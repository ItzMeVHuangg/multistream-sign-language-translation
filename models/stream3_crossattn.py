
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as tv_video
from typing import Optional, Tuple

def interpolate_temporal(x: torch.Tensor, target_T: int) -> torch.Tensor:
    if x.shape[1] == target_T:
        return x
    x = x.transpose(1, 2)                      
    x = F.interpolate(x, size=target_T, mode='linear', align_corners=False)
    return x.transpose(1, 2)                   

class SwinPatchExtractor(nn.Module):
    def __init__(
        self,
        pretrained:      bool = True,
        use_checkpoint:  bool = True,
        _swin_ref=None,         
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        if _swin_ref is not None:
            self.patch_embed = _swin_ref.swin_patch_embed
            self.pos_drop    = _swin_ref.swin_pos_drop
            self.stage1      = _swin_ref.swin_stage1
            self.merge1      = _swin_ref.swin_merge1
            self.stage2      = _swin_ref.swin_stage2
            self.merge2      = _swin_ref.swin_merge2
            self.stage3      = _swin_ref.swin_stage3
            self.merge3      = _swin_ref.swin_merge3
            self.stage4      = _swin_ref.swin_stage4
            self.norm        = _swin_ref.swin_norm
        else:
            weights = tv_video.Swin3D_T_Weights.DEFAULT if pretrained else None
            swin    = tv_video.swin3d_t(weights=weights)
            self.patch_embed = swin.patch_embed
            self.pos_drop    = swin.pos_drop
            self.stage1      = swin.features[0]
            self.merge1      = swin.features[1]
            self.stage2      = swin.features[2]
            self.merge2      = swin.features[3]
            self.stage3      = swin.features[4]
            self.merge3      = swin.features[5]
            self.stage4      = swin.features[6]
            self.norm        = swin.norm
            del swin

    def _ckpt(self, module, x):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        x = frames.permute(0, 2, 1, 3, 4).contiguous() 

        x = self.patch_embed(x)   
        x = self.pos_drop(x)
        x = self._ckpt(self.stage1, x)
        x = self._ckpt(self.merge1, x)
        x = self._ckpt(self.stage2, x)
        x = self._ckpt(self.merge2, x)
        x = self._ckpt(self.stage3, x)
        x = self._ckpt(self.merge3, x)
        x = self._ckpt(self.stage4, x)  
        x = self.norm(x)

        B2, T2, Hp, Wp, C2 = x.shape
        x = x.reshape(B2, T2, Hp * Wp, C2)
        return x

class LandmarkVisualCrossAttention(nn.Module):
    def __init__(
        self,
        d_model:  int   = 512,
        swin_dim: int   = 768,
        nhead:    int   = 8,
        dropout:  float = 0.1,
    ):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead   = nhead
        self.d_head  = d_model // nhead
        self.scale   = self.d_head ** -0.5

        self.k_proj  = nn.Linear(swin_dim, d_model, bias=False)
        self.v_proj  = nn.Linear(swin_dim, d_model, bias=False)
        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop    = nn.Dropout(p=dropout)
        self.norm    = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,    
        kv:    torch.Tensor,  
    ) -> torch.Tensor:
        B, T, _ = query.shape
        P = kv.shape[2]

        Q = self.q_proj(query)                        
        K = self.k_proj(kv.reshape(B * T, P, -1))     
        V = self.v_proj(kv.reshape(B * T, P, -1))    

        Q = Q.reshape(B * T, 1, self.nhead, self.d_head).transpose(1, 2)  
        K = K.reshape(B * T, P, self.nhead, self.d_head).transpose(1, 2) 
        V = V.reshape(B * T, P, self.nhead, self.d_head).transpose(1, 2) 

        attn = (Q @ K.transpose(-2, -1)) * self.scale    
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = (attn @ V)                           
        out = out.transpose(1, 2).reshape(B * T, 1, -1).squeeze(1)
        out = self.out_proj(out).reshape(B, T, -1)  

        out = self.norm(out + query)
        return out


class CTCHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.norm   = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(p=dropout)
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (T, B, num_classes)"""
        x = self.drop(self.norm(x))
        log_probs = F.log_softmax(self.linear(x), dim=-1)
        return log_probs.permute(1, 0, 2)


class Stream3_CrossAttn(nn.Module):
   
    _SWIN_PATCH_DIM = 768

    def __init__(
        self,
        keypoint_dim:    int   = 225,
        d_model:         int   = 512,
        nhead:           int   = 8,
        bilstm_hidden:   int   = 512,
        bilstm_layers:   int   = 2,
        num_classes:     int   = 1296,
        attn_dropout:    float = 0.1,
        bilstm_dropout:  float = 0.3,
        ctc_dropout:     float = 0.1,
        pretrained:      bool  = True,
        use_checkpoint:  bool  = True,
        swin_ref=None,
    ):
        super().__init__()

        self.swin = SwinPatchExtractor(
            pretrained     = pretrained,
            use_checkpoint = use_checkpoint,
            _swin_ref      = swin_ref,
        )
        swin_dim = getattr(swin_ref, 'swin_out_dim', self._SWIN_PATCH_DIM)

        self.landmark_proj = nn.Sequential(
            nn.Linear(keypoint_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(p=0.1),
        )

        self.cross_attn = LandmarkVisualCrossAttention(
            d_model  = d_model,
            swin_dim = swin_dim,
            nhead    = nhead,
            dropout  = attn_dropout,
        )

        self.bilstm = nn.LSTM(
            input_size    = d_model,
            hidden_size   = bilstm_hidden,
            num_layers    = bilstm_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = bilstm_dropout if bilstm_layers > 1 else 0.0,
        )
        bilstm_out = bilstm_hidden * 2

        self.lstm_norm = nn.LayerNorm(bilstm_out)
        self.lstm_drop = nn.Dropout(p=bilstm_dropout)

        self.lstm_proj = nn.Sequential(
            nn.Linear(bilstm_out, d_model),
            nn.GELU(),
            nn.Dropout(p=0.1),
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.d_model  = d_model

        self.ctc_head = CTCHead(d_model, num_classes, dropout=ctc_dropout)

    def forward(
        self,
        frames:     torch.Tensor,    
        keypoints:  torch.Tensor,    
        frame_lens: torch.Tensor,   
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      
        B, T, C, H, W = frames.shape
        patch_feats = self.swin(frames)
        T_prime = patch_feats.shape[1]

        landmark_q = self.landmark_proj(keypoints)   
        landmark_q = interpolate_temporal(landmark_q, T_prime)  
        grounded = self.cross_attn(landmark_q, patch_feats) 

        scale    = T_prime / max(T, 1)
        out_lens = (frame_lens.float() * scale).long().clamp(min=1, max=T_prime)

        packed = nn.utils.rnn.pack_padded_sequence(
            grounded, out_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.bilstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        lstm_out = self.lstm_norm(lstm_out)
        lstm_out = self.lstm_drop(lstm_out)
        feats    = self.lstm_proj(lstm_out)  
        feats    = self.out_norm(feats)

        log_probs = self.ctc_head(feats)    
        return log_probs, feats, out_lens
if __name__ == "__main__":
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, T, C, H, W = 2, 32, 3, 224, 224
    KD             = 225
    num_classes    = 1296

    model = Stream3_CrossAttn(
        keypoint_dim   = KD,
        d_model        = 512,
        nhead          = 8,
        bilstm_hidden  = 512,
        bilstm_layers  = 2,
        num_classes    = num_classes,
        pretrained     = False,
        use_checkpoint = False,
    ).to(device)

    n_total  = sum(p.numel() for p in model.parameters())
    n_swin   = sum(p.numel() for p in model.swin.parameters())
    n_other  = n_total - n_swin
    print(f"Stream3 CrossAttn | total params: {n_total:,}")
    print(f"  Swin-T  : {n_swin:,}")
    print(f"  Rest    : {n_other:,}")

    frames     = torch.randn(B, T, C, H, W, device=device)
    keypoints  = torch.randn(B, T, KD, device=device)
    frame_lens = torch.tensor([T, T // 2], device=device)

    t0 = time.time()
    with torch.no_grad():
        log_probs, feats, out_lens = model(frames, keypoints, frame_lens)
    elapsed = time.time() - t0

    print(f"\n  Input frames  : {list(frames.shape)}")
    print(f"  Input kpts    : {list(keypoints.shape)}")
    print(f"  log_probs     : {list(log_probs.shape)}  (T', B, num_classes)")
    print(f"  feats         : {list(feats.shape)}       (B, T', d_model)")
    print(f"  out_lens      : {out_lens.tolist()}")
    print(f"  Time          : {elapsed:.3f}s")
