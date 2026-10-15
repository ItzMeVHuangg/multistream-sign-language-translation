
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as tv_video
from typing import Tuple

class TAPEAdapter(nn.Module):
    def __init__(
        self,
        channels:    int,
        kernel_size: int   = 3,
        dilation:    int   = 1,
        max_len:     int   = 512,
        dropout:     float = 0.1,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2

        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, channels))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.norm = nn.LayerNorm(channels)
        self.dw_conv = nn.Conv1d(
            channels, channels,
            kernel_size = kernel_size,
            dilation    = dilation,
            padding     = pad,
            groups      = channels,
            bias        = False,
        )
        nn.init.zeros_(self.dw_conv.weight)  

        self.act  = nn.GELU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        residual = x

        pos = self.pos_emb[:, :T, :]           
        x   = self.norm(x + pos)               
        x   = x.transpose(1, 2)             
        x   = self.act(self.dw_conv(x))      
        x   = self.drop(x)
        x   = x.transpose(1, 2)               
        return x + residual

class CTCHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.norm   = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(p=dropout)
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.norm(x))
        logits    = self.linear(x)                
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.permute(1, 0, 2)        


class MultiScaleTemporalConv(nn.Module):
    def __init__(
        self,
        d_model: int,
        kernels=(3, 5, 7),
        dilations=(1, 2, 4),
        dropout: float = 0.1,
    ):
        super().__init__()
        branches = []
        for kernel in kernels:
            for dilation in dilations:
                pad = (kernel - 1) * dilation // 2
                branches.append(
                    nn.Sequential(
                        nn.Conv1d(
                            d_model,
                            d_model,
                            kernel_size=kernel,
                            dilation=dilation,
                            padding=pad,
                            groups=min(d_model, 8),
                            bias=False,
                        ),
                        nn.BatchNorm1d(d_model),
                        nn.GELU(),
                    )
                )
        self.branches = nn.ModuleList(branches)
        self.proj = nn.Sequential(
            nn.Conv1d(d_model * len(branches), d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        xt = x.transpose(1, 2)
        out = torch.cat([branch(xt) for branch in self.branches], dim=1)
        out = self.proj(out).transpose(1, 2)
        return self.norm(out + residual)


class Stream1_RGB_USTM(nn.Module):

    _TAPE_SHORT_DIM = 96      
    _TAPE_LONG_DIM  = 384   
    _SWIN_FINAL_DIM = 768

    def __init__(
        self,
        pretrained:           bool  = True,
        num_classes:          int   = 1296,
        d_model:              int   = 512,
        tape_kernel:          int   = 3,
        tape_dilation_short:  int   = 1,
        tape_dilation_long:   int   = 3,
        tape_max_len:         int   = 512,
        tape_dropout:         float = 0.1,
        temporal_dropout:     float = 0.1,
        temporal_lstm_layers: int   = 1,
        proj_dropout:         float = 0.1,
        ctc_dropout:          float = 0.1,
        use_checkpoint:       bool  = True,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        weights = tv_video.Swin3D_T_Weights.DEFAULT if pretrained else None
        swin    = tv_video.swin3d_t(weights=weights)

        self.swin_patch_embed = swin.patch_embed   
        self.swin_pos_drop    = swin.pos_drop
        self.swin_stage1      = swin.features[0]   
        self.swin_merge1      = swin.features[1]   
        self.swin_stage2      = swin.features[2]   
        self.swin_merge2      = swin.features[3]   
        self.swin_stage3      = swin.features[4]   
        self.swin_merge3      = swin.features[5]   
        self.swin_stage4      = swin.features[6]   
        self.swin_norm        = swin.norm           

        del swin 
        self.tape_short = TAPEAdapter(
            channels    = self._TAPE_SHORT_DIM,
            kernel_size = tape_kernel,
            dilation    = tape_dilation_short,
            max_len     = tape_max_len,
            dropout     = tape_dropout,
        )
        self.tape_long = TAPEAdapter(
            channels    = self._TAPE_LONG_DIM,
            kernel_size = tape_kernel,
            dilation    = tape_dilation_long,
            max_len     = tape_max_len,
            dropout     = tape_dropout,
        )

        self.proj = nn.Sequential(
            nn.Linear(self._SWIN_FINAL_DIM, d_model),
            nn.GELU(),
            nn.Dropout(p=proj_dropout),
        )
        self.out_norm   = nn.LayerNorm(d_model)
        self.d_model    = d_model
        self.ms_temporal = MultiScaleTemporalConv(
            d_model=d_model,
            dropout=temporal_dropout,
        )
        self.temporal_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=temporal_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=temporal_dropout if temporal_lstm_layers > 1 else 0.0,
        ) if temporal_lstm_layers > 0 else None
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_drop = nn.Dropout(temporal_dropout)

        self.ctc_head = CTCHead(d_model, num_classes, dropout=ctc_dropout)

    def _ckpt(self, module, x):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                module, x, use_reentrant=False
            )
        return module(x)

    @staticmethod
    def _spatial_mean_nhwc(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3))   

    def _apply_tape(
        self,
        x:    torch.Tensor, 
        tape: TAPEAdapter,
    ) -> torch.Tensor:
        
        avg       = self._spatial_mean_nhwc(x)    
        tape_out  = tape(avg)                      
        delta     = (tape_out - avg).unsqueeze(2).unsqueeze(3) 
        return x + delta

    def forward(
        self,
        frames:     torch.Tensor,  
        frame_lens: torch.Tensor, 
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B, T, C, H, W = frames.shape
        x = frames.permute(0, 2, 1, 3, 4).contiguous() 

        x = self.swin_patch_embed(x)    
        x = self.swin_pos_drop(x)

        x = self._ckpt(self.swin_stage1, x)  

        x = self._apply_tape(x, self.tape_short)  

        x = self._ckpt(self.swin_merge1, x)   
        x = self._ckpt(self.swin_stage2, x)    

        x = self._ckpt(self.swin_merge2, x)   
        x = self._ckpt(self.swin_stage3, x)   

        x = self._apply_tape(x, self.tape_long)   

        x = self._ckpt(self.swin_merge3, x)    
        x = self._ckpt(self.swin_stage4, x)    

        x = self.swin_norm(x)                  
        feats = self._spatial_mean_nhwc(x)    

        feats = self.proj(feats)               
        feats = self.out_norm(feats)

        T_prime  = feats.shape[1]
        scale    = T_prime / max(T, 1)
        out_lens = (frame_lens.float() * scale).long().clamp(min=1, max=T_prime)

        feats = self.ms_temporal(feats)

        if self.temporal_lstm is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                feats,
                out_lens.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            lstm_out, _ = self.temporal_lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out,
                batch_first=True,
                total_length=T_prime,
            )
            feats = self.temporal_norm(feats + self.temporal_drop(lstm_out))

        log_probs = self.ctc_head(feats)       
        return log_probs, feats, out_lens

if __name__ == "__main__":
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, T, C, H, W = 2, 32, 3, 224, 224
    num_classes    = 1296

    model = Stream1_RGB_USTM(
        pretrained          = False,
        num_classes         = num_classes,
        d_model             = 512,
        tape_kernel         = 3,
        tape_dilation_short = 1,
        tape_dilation_long  = 3,
        use_checkpoint      = False,
    ).to(device)

    n_total    = sum(p.numel() for p in model.parameters())
    n_tape     = sum(p.numel() for p in model.tape_short.parameters()) + \
                 sum(p.numel() for p in model.tape_long.parameters())
    n_swin     = n_total - n_tape - \
                 sum(p.numel() for p in model.proj.parameters()) - \
                 sum(p.numel() for p in model.ctc_head.parameters()) - \
                 sum(p.numel() for p in model.out_norm.parameters())

    print(f"Stream1 RGB-USTM")
    print(f"  Total params : {n_total:,}")
    print(f"  Swin-T params: {n_swin:,}")
    print(f"  TAPE params  : {n_tape:,}  ({100*n_tape/n_total:.2f}% of total)")

    frames     = torch.randn(B, T, C, H, W, device=device)
    frame_lens = torch.tensor([T, T // 2], device=device)

    t0 = time.time()
    with torch.no_grad():
        log_probs, feats, out_lens = model(frames, frame_lens)
    elapsed = time.time() - t0

    print(f"\n  Input  frames : {list(frames.shape)}")
    print(f"  log_probs     : {list(log_probs.shape)}  (T', B, num_classes)")
    print(f"  feats         : {list(feats.shape)}       (B, T', d_model)")
    print(f"  out_lens      : {out_lens.tolist()}")
    print(f"  Time (no_grad): {elapsed:.3f}s")
