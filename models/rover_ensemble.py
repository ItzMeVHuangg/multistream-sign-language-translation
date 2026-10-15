import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np


class GlossBigramLM(nn.Module):
 

    def __init__(self, vocab_size: int, smoothing: float = 1e-3):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing  = smoothing
        self.log_transition = nn.Parameter(
            torch.zeros(vocab_size, vocab_size)
        )
        nn.init.uniform_(self.log_transition, -0.1, 0.1)

    def get_log_prob(self, prev_id: int, next_id: int) -> float:
        with torch.no_grad():
            lp = F.log_softmax(self.log_transition[prev_id], dim=-1)
            return lp[next_id].item()

    def score_sequence(self, gloss_ids: List[int]) -> float:
        if len(gloss_ids) < 2:
            return 0.0
        score = 0.0
        with torch.no_grad():
            lp_mat = F.log_softmax(self.log_transition, dim=-1)
            for i in range(len(gloss_ids) - 1):
                score += lp_mat[gloss_ids[i], gloss_ids[i + 1]].item()
        return score

    def forward(self, prev_ids: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.log_transition[prev_ids], dim=-1)


def _levenshtein_alignment(
    seq_a: List[int],
    seq_b: List[int],
) -> List[Tuple[Optional[int], Optional[int]]]:
    M, N = len(seq_a), len(seq_b)
    dp   = [[0] * (N + 1) for _ in range(M + 1)]
    back = [[None] * (N + 1) for _ in range(M + 1)]

    for i in range(M + 1):
        dp[i][0] = i
        back[i][0] = ('del', i - 1, 0) if i > 0 else None
    for j in range(N + 1):
        dp[0][j] = j
        back[0][j] = ('ins', 0, j - 1) if j > 0 else None

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            choices = [
                (dp[i - 1][j - 1] + cost, 'sub', i - 1, j - 1),
                (dp[i - 1][j]     + 1,    'del', i - 1, j),
                (dp[i][j - 1]     + 1,    'ins', i,     j - 1),
            ]
            best = min(choices, key=lambda x: x[0])
            dp[i][j] = best[0]
            back[i][j] = best[1:]

    path = []
    i, j = M, N
    while i > 0 or j > 0:
        if back[i][j] is None:
            break
        op, pi, pj = back[i][j]
        if op == 'sub':
            path.append((pi, pj))
            i, j = pi, pj
        elif op == 'del':
            path.append((pi, None))
            i, j = pi, j
        else:
            path.append((None, pj))
            i, j = i, pj

    path.reverse()
    return path


def _build_wtn(
    sequences:   List[List[int]],
    confidences: List[List[float]],
    blank_idx:   int = 0,
) -> List[Dict]:
  
    if not sequences:
        return []

    wtn_slots = [
        {
            'candidates': [
                (sequences[0][i], confidences[0][i] if i < len(confidences[0]) else 1.0)
            ],
            'voters': 1,
        }
        for i in range(len(sequences[0]))
    ]

    for s_idx in range(1, len(sequences)):
        seq  = sequences[s_idx]
        conf = confidences[s_idx]
        wtn_hyp = [
            max(slot['candidates'], key=lambda x: x[1])[0]
            for slot in wtn_slots
        ]
        alignment = _levenshtein_alignment(wtn_hyp, seq)
        new_slots = []
        for (a_idx, b_idx) in alignment:
            if a_idx is not None and b_idx is not None:
                slot = dict(wtn_slots[a_idx])
                slot['candidates'] = list(slot['candidates'])
                g   = seq[b_idx]
                c   = conf[b_idx] if b_idx < len(conf) else 1.0
                slot['candidates'].append((g, c))
                slot['voters'] = slot.get('voters', s_idx) + 1
                new_slots.append(slot)
            elif a_idx is not None:
                slot = dict(wtn_slots[a_idx])
                slot['candidates'] = list(slot['candidates'])
                slot['candidates'].append((blank_idx, 1.0))
                slot['voters'] = slot.get('voters', s_idx) + 1
                new_slots.append(slot)
            else:
                g = seq[b_idx]
                c = conf[b_idx] if b_idx < len(conf) else 1.0
                new_slots.append({
                    'candidates': [(blank_idx, 1.0)] * s_idx + [(g, c)],
                    'voters': s_idx + 1,
                })

        wtn_slots = new_slots

    return wtn_slots


def _vote_wtn(
    wtn_slots:   List[Dict],
    blank_idx:   int = 0,
    lm:          Optional[GlossBigramLM] = None,
    lm_weight:   float = 0.3,
) -> List[int]:
    result = []
    prev_id = None

    for slot in wtn_slots:
        scores: Dict[int, float] = {}
        for gloss_id, conf in slot['candidates']:
            scores[gloss_id] = scores.get(gloss_id, 0.0) + conf

        if not scores:
            continue

        if lm is not None and prev_id is not None:
            scores = {
                g: s + lm_weight * lm.get_log_prob(prev_id, g)
                for g, s in scores.items()
            }

        best_gloss = max(scores, key=scores.__getitem__)

        if best_gloss == blank_idx:
            continue

        result.append(best_gloss)
        prev_id = best_gloss

    return result

def ctc_beam_with_scores(
    log_probs:  torch.Tensor,   # (T, C)
    beam_width: int   = 10,
    blank_idx:  int   = 0,
) -> Tuple[List[int], List[float]]:

    import math
    T, C = log_probs.shape
    probs = log_probs.exp().cpu().numpy()
    NEG_INF = float('-inf')

    def lse(a, b):
        if a == NEG_INF: return b
        if b == NEG_INF: return a
        return max(a, b) + math.log1p(math.exp(-abs(a - b)))
    beams = {(): (0.0, NEG_INF)}

    for t in range(T):
        new_beams: Dict = {}
        for prefix, (pb, pnb) in beams.items():
            p_total = lse(pb, pnb)
            for c in range(C):
                lp = math.log(float(probs[t, c]) + 1e-30)
                if c == blank_idx:
                    key = prefix
                    old = new_beams.get(key, (NEG_INF, NEG_INF))
                    new_beams[key] = (lse(old[0], p_total + lp), old[1])
                else:
                    key = prefix + (c,)
                    old = new_beams.get(key, (NEG_INF, NEG_INF))
                    if prefix and prefix[-1] == c:
                        new_beams[key] = (old[0], lse(old[1], pb + lp))
                    else:
                        new_beams[key] = (old[0], lse(old[1], p_total + lp))

        beams = dict(
            sorted(new_beams.items(),
                   key=lambda x: lse(x[1][0], x[1][1]),
                   reverse=True)[:beam_width]
        )

    best_prefix, (pb, pnb) = max(
        beams.items(), key=lambda x: lse(x[1][0], x[1][1])
    )
    gloss_ids = list(best_prefix)
    confidences = [1.0] * len(gloss_ids)

    return gloss_ids, confidences

class ROVEREnsemble(nn.Module):
   
    def __init__(
        self,
        vocab_size: int,
        blank_idx:  int   = 0,
        beam_width: int   = 10,
        lm_weight:  float = 0.3,
        use_lm:     bool  = True,
        stream_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.blank_idx  = blank_idx
        self.beam_width = beam_width
        self.lm_weight  = lm_weight
        self.stream_weights = stream_weights or [1.0, 1.0, 1.0]

        self.lm = GlossBigramLM(vocab_size) if use_lm else None

    @torch.no_grad()
    def decode_stream(
        self,
        log_probs: torch.Tensor,   # (T, B, C)
        lengths:   torch.Tensor,   # (B,)
    ) -> Tuple[List[List[int]], List[List[float]]]:
        from utils.ctc_decoder import ctc_greedy_decode

        T, B, C = log_probs.shape
        all_seqs, all_confs = [], []
        for b in range(B):
            L  = lengths[b].item()
            lp = log_probs[:L, b, :]                # (L, C)
            if self.beam_width <= 1:
                seq = ctc_greedy_decode(lp, blank_idx=self.blank_idx)[0]
                conf = [1.0] * len(seq)
            else:
                seq, conf = ctc_beam_with_scores(
                    lp, beam_width=self.beam_width, blank_idx=self.blank_idx
                )
            all_seqs.append(seq)
            all_confs.append(conf)
        return all_seqs, all_confs

    @torch.no_grad()
    def forward(
        self,
        log_probs_list: List[torch.Tensor],   # 3 × (T, B, C)
        lengths_list:   List[torch.Tensor],   # 3 × (B,)
    ) -> List[List[int]]:
        assert len(log_probs_list) == 3
        B = log_probs_list[0].shape[1]
        active = [i for i, weight in enumerate(self.stream_weights) if weight > 0.0]
        if not active:
            active = [0]
        stream_seqs  = []   # 3 × [B × seq]
        stream_confs = []
        for i in active:
            lp, le = log_probs_list[i], lengths_list[i]
            seqs, confs = self.decode_stream(lp, le)
            stream_seqs.append(seqs)
            stream_confs.append(confs)
        final_seqs = []
        for b in range(B):
            seqs  = [stream_seqs[j][b]  for j in range(len(active))]
            confs = [
                [self.stream_weights[i] * c for c in stream_confs[j][b]]
                for j, i in enumerate(active)
            ]

            wtn    = _build_wtn(seqs, confs, blank_idx=self.blank_idx)
            result = _vote_wtn(wtn, self.blank_idx, self.lm, self.lm_weight)
            final_seqs.append(result)

        return final_seqs

class ThreeStreamCSLR(nn.Module):
    
    def __init__(
        self,
        num_classes: int,
        cfg:         dict,
        share_swin:  bool = True,
    ):
        super().__init__()
        from models.stream1_rgb_ustm   import Stream1_RGB_USTM
        from models.stream2_hypergraph import Stream2_Hypergraph
        from models.stream3_crossattn  import Stream3_CrossAttn

        s1_cfg = cfg.get('stream1', {})
        s2_cfg = cfg.get('stream2', {})
        s3_cfg = cfg.get('stream3', {})
        ens    = cfg.get('ensemble', {})

        self.stream1 = Stream1_RGB_USTM(
            pretrained          = s1_cfg.get('pretrained', True),
            num_classes         = num_classes,
            d_model             = s1_cfg.get('d_model', 1024),
            tape_kernel         = s1_cfg.get('tape_kernel', 3),
            tape_dilation_short = s1_cfg.get('tape_dilation_short', 1),
            tape_dilation_long  = s1_cfg.get('tape_dilation_long', 3),
            tape_max_len        = s1_cfg.get('tape_max_len', 512),
            tape_dropout        = s1_cfg.get('tape_dropout', 0.1),
            temporal_dropout    = s1_cfg.get('temporal_dropout', 0.1),
            temporal_lstm_layers = s1_cfg.get('temporal_lstm_layers', 1),
            ctc_dropout         = s1_cfg.get('ctc_dropout', 0.1),
            use_checkpoint      = s1_cfg.get('use_checkpoint', True),
            backbone            = s1_cfg.get('backbone', 'swint'),
            temporal_head       = s1_cfg.get('temporal_head', 'mstcn'),
            conv_type           = s1_cfg.get('conv_type', 3),
            adapter             = s1_cfg.get('adapter', 3),
            hidden_size         = s1_cfg.get('hidden_size', s1_cfg.get('d_model', 1024)),
            weight_norm         = s1_cfg.get('weight_norm', True),
            share_classifier    = s1_cfg.get('share_classifier', True),
        )

        self.stream2 = Stream2_Hypergraph(
            keypoint_dim = s2_cfg.get('keypoint_dim', 225),
            num_joints   = s2_cfg.get('num_joints', 75),
            keypoint_layout = s2_cfg.get('keypoint_layout', 'pose_hands'),
            node_dim     = s2_cfg.get('node_dim', 128),
            d_model      = s2_cfg.get('d_model', 512),
            num_proto    = s2_cfg.get('num_proto', 8),
            knn_k        = s2_cfg.get('knn_k', 8),
            ms_kernels   = s2_cfg.get('ms_kernels', [1, 3, 5, 7]),
            dropout      = s2_cfg.get('dropout', 0.1),
            ctc_dropout  = s2_cfg.get('ctc_dropout', 0.1),
            normalize_keypoints = s2_cfg.get('normalize_keypoints', True),
            temporal_lstm_layers = s2_cfg.get('temporal_lstm_layers', 1),
            cgp_channels = s2_cfg.get('cgp_channels', [64, 128, 128, 256]),
            num_classes  = num_classes,
        )

        swin_ref = self.stream1 if share_swin else None
        self.stream3 = Stream3_CrossAttn(
            keypoint_dim   = s3_cfg.get('keypoint_dim', 225),
            d_model        = s3_cfg.get('d_model', 512),
            nhead          = s3_cfg.get('nhead', 8),
            bilstm_hidden  = s3_cfg.get('bilstm_hidden', 512),
            bilstm_layers  = s3_cfg.get('bilstm_layers', 2),
            num_classes    = num_classes,
            attn_dropout   = s3_cfg.get('attn_dropout', 0.1),
            bilstm_dropout = s3_cfg.get('bilstm_dropout', 0.3),
            pretrained     = s3_cfg.get('pretrained', True),
            use_checkpoint = s3_cfg.get('use_checkpoint', True),
            swin_ref       = swin_ref,
        )

        self.rover = ROVEREnsemble(
            vocab_size  = num_classes,
            blank_idx   = cfg.get('cslr', {}).get('ctc_blank_idx', 0),
            beam_width  = ens.get('beam_width', 10),
            lm_weight   = ens.get('lm_weight', 0.3),
            use_lm      = ens.get('use_lm', True),
            stream_weights = ens.get('stream_weights', [1.0, 1.0, 1.0]),
        )
        init_weights = torch.tensor(
            ens.get('stream_weights', [1.0, 1.0, 1.0]),
            dtype=torch.float32,
        ).clamp_min(1e-3)
        self.fusion_logits = nn.Parameter(init_weights.log())
        self.fusion_log_temps = nn.Parameter(torch.zeros(3))
        self.fusion_blank_bias = nn.Parameter(torch.zeros(3))
        self.blank_idx = cfg.get('cslr', {}).get('ctc_blank_idx', 0)

    @staticmethod
    def _resize_log_probs(log_probs: torch.Tensor, target_t: int) -> torch.Tensor:
        if log_probs.shape[0] == target_t:
            return log_probs
        probs = log_probs.exp().permute(1, 2, 0)  # (B, C, T)
        probs = F.interpolate(probs, size=target_t, mode='linear', align_corners=False)
        return probs.permute(2, 0, 1).clamp_min(1e-8).log()

    def _fuse_log_probs(
        self,
        log_probs_list: List[torch.Tensor],
        lengths_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_t = min(lp.shape[0] for lp in log_probs_list)
        weights = F.softmax(self.fusion_logits[:len(log_probs_list)], dim=0)
        temperatures = self.fusion_log_temps[:len(log_probs_list)].exp().clamp(0.5, 2.0)
        blank_biases = self.fusion_blank_bias[:len(log_probs_list)].clamp(-5.0, 5.0)

        calibrated = []
        scaled_lengths = []
        for temp, blank_bias, log_probs, lengths in zip(
            temperatures, blank_biases, log_probs_list, lengths_list
        ):
            resized = self._resize_log_probs(log_probs, target_t)
            adjusted = resized / temp
            adjusted = adjusted.clone()
            adjusted[:, :, self.blank_idx] = adjusted[:, :, self.blank_idx] + blank_bias
            calibrated.append(F.log_softmax(adjusted, dim=-1))
            scale = target_t / max(1, log_probs.shape[0])
            scaled_lengths.append(
                torch.ceil(lengths.float() * scale).long().clamp(min=1, max=target_t)
            )

        fused_lens = torch.stack(scaled_lengths, dim=0).min(dim=0).values
        stacked = torch.stack(
            [lp + weights[i].clamp_min(1e-8).log() for i, lp in enumerate(calibrated)],
            dim=0,
        )
        fused_log_probs = torch.logsumexp(stacked, dim=0)
        return fused_log_probs, fused_lens, weights

    def forward(
        self,
        frames:     torch.Tensor,   
        keypoints:  torch.Tensor,   
        frame_lens: torch.Tensor,   
    ) -> dict:
        lp1, f1, l1 = self.stream1(frames, frame_lens)
        lp2, f2, l2 = self.stream2(keypoints, frame_lens)
        lp3, f3, l3 = self.stream3(frames, keypoints, frame_lens)
        fused_lp, fused_lens, fusion_weights = self._fuse_log_probs(
            [lp1, lp2, lp3],
            [l1, l2, l3],
        )

        return {
            's1_log_probs': lp1, 's1_feats': f1, 's1_lens': l1,
            's2_log_probs': lp2, 's2_feats': f2, 's2_lens': l2,
            's3_log_probs': lp3, 's3_feats': f3, 's3_lens': l3,
            'fused_log_probs': fused_lp, 'fused_lens': fused_lens,
            'fusion_weights': fusion_weights,
            'fusion_temperatures': self.fusion_log_temps[:3].detach().exp().clamp(0.5, 2.0),
            'fusion_blank_biases': self.fusion_blank_bias[:3].detach().clamp(-5.0, 5.0),
        }

    @torch.no_grad()
    def predict(
        self,
        frames:     torch.Tensor,
        keypoints:  torch.Tensor,
        frame_lens: torch.Tensor,
    ) -> List[List[int]]:
        out = self.forward(frames, keypoints, frame_lens)
        return self.rover(
            [out['s1_log_probs'], out['s2_log_probs'], out['s3_log_probs']],
            [out['s1_lens'],      out['s2_lens'],      out['s3_lens']],
        )
