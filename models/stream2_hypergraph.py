
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

def _build_physical_edges(num_joints: int = 75) -> List[Tuple[int, ...]]:
    edges = []
    lh_fingers = [
        (0,1,2,3,4), (0,5,6,7,8), (0,9,10,11,12),
        (0,13,14,15,16), (0,17,18,19,20),
        (0,5,9,13,17),   # palm arch
    ]
    for f in lh_fingers:
        edges.append(tuple(f))           
        edges.append(tuple(j+21 for j in f)) 

    left_arm  = (42+11, 42+13, 42+15)   
    right_arm = (42+12, 42+14, 42+16)
    edges.append(left_arm)
    edges.append(right_arm)

    edges.append((42+15, 42+16))        
    edges.append((42+13, 42+14))        
    edges.append((42+11, 42+12))       

    return edges


def _incidence_matrix(edges: List[Tuple[int, ...]], num_nodes: int) -> torch.Tensor:
    E = len(edges)
    H = torch.zeros(num_nodes, E)
    for e_idx, edge in enumerate(edges):
        for v in edge:
            if v < num_nodes:
                H[v, e_idx] = 1.0
    return H  


def _build_physical_adjacency(num_joints: int = 75) -> torch.Tensor:
    A = torch.eye(num_joints)
    for edge in _build_physical_edges(num_joints):
        for u, v in zip(edge[:-1], edge[1:]):
            if u < num_joints and v < num_joints:
                A[u, v] = 1.0
                A[v, u] = 1.0
    deg = A.sum(dim=1).clamp(min=1.0)
    deg_inv_sqrt = deg.pow(-0.5)
    return deg_inv_sqrt[:, None] * A * deg_inv_sqrt[None, :]


class HypergraphConv(nn.Module):
   
    def __init__(self, in_dim: int, out_dim: int, num_edges: int, dropout: float = 0.1):
        super().__init__()
        self.theta   = nn.Linear(in_dim, out_dim, bias=False)
        self.w_diag  = nn.Parameter(torch.ones(num_edges))
        self.norm    = nn.LayerNorm(out_dim)
        self.act     = nn.GELU()
        self.drop    = nn.Dropout(p=dropout)

        self.res_proj = nn.Linear(in_dim, out_dim, bias=False) \
            if in_dim != out_dim else nn.Identity()

    def forward(
        self,
        X: torch.Tensor,     
        H: torch.Tensor,     
    ) -> torch.Tensor:
        B, T, N, C = X.shape
        X_flat = X.reshape(B * T, N, C)        

        X_theta = self.theta(X_flat)
        H = H.to(X.device)

        if H.dim() == 2:
            E = H.shape[1]
            D_e = H.sum(0).clamp(min=1e-6)
            D_v = H.sum(1).clamp(min=1e-6)
            W = F.softplus(self.w_diag[:E])
            W_norm = W / D_e
            agg_e = torch.einsum('en,bnd->bed', H.t(), X_theta)
            agg_e = agg_e * W_norm.unsqueeze(0).unsqueeze(-1)
            agg_v = torch.einsum('ne,bed->bnd', H, agg_e)
            agg_v = agg_v / D_v.unsqueeze(0).unsqueeze(-1)
        elif H.dim() == 3:
            E = H.shape[2]
            if H.shape[0] != B * T:
                raise ValueError(f"Dynamic H batch mismatch: got {H.shape[0]}, expected {B * T}")
            D_e = H.sum(1).clamp(min=1e-6)
            D_v = H.sum(2).clamp(min=1e-6)
            W = F.softplus(self.w_diag[:E])
            W_norm = W.unsqueeze(0) / D_e
            agg_e = torch.einsum('bne,bnd->bed', H, X_theta)
            agg_e = agg_e * W_norm.unsqueeze(-1)
            agg_v = torch.einsum('bne,bed->bnd', H, agg_e)
            agg_v = agg_v / D_v.unsqueeze(-1)
        else:
            raise ValueError(f"Expected H with shape (N,E) or (B*T,N,E), got {tuple(H.shape)}")

        out = self.act(self.norm(agg_v))
        out = self.drop(out)

        out = out + self.res_proj(X_flat)

        return out.reshape(B, T, N, -1)        


class GraphConvBranch(nn.Module):
    def __init__(self, num_nodes: int, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("A", _build_physical_adjacency(num_nodes))
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout)
        self.res_proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, T, N, C = X.shape
        x = X.reshape(B * T, N, C)
        out = torch.einsum("nm,bmc->bnc", self.A.to(X.device), self.proj(x))
        out = self.drop(self.act(self.norm(out)))
        out = out + self.res_proj(x)
        return out.reshape(B, T, N, -1)


class HyperSignLiteCGPBlock(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        out_dim: int,
        num_proto: int = 8,
        knn_k: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        branch_dim = max(out_dim // 2, 32)
        self.num_nodes = num_nodes
        self.num_proto = num_proto
        self.knn_k = knn_k
        self.low_proj = nn.Sequential(nn.Linear(in_dim, branch_dim), nn.LayerNorm(branch_dim), nn.GELU())
        self.high_proj = nn.Sequential(nn.Linear(in_dim, branch_dim), nn.LayerNorm(branch_dim), nn.GELU())
        self.gcn = GraphConvBranch(num_nodes, branch_dim, branch_dim, dropout)
        self.hyper = HypergraphConv(branch_dim, branch_dim, num_edges=num_nodes + num_proto, dropout=dropout)
        self.proto = nn.Parameter(torch.empty(num_proto, branch_dim))
        nn.init.trunc_normal_(self.proto, std=0.02)
        self.gate = nn.Sequential(
            nn.Linear(branch_dim * 2, branch_dim),
            nn.GELU(),
            nn.Linear(branch_dim, branch_dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(nn.Linear(branch_dim, out_dim), nn.GELU(), nn.Dropout(p=dropout))
        self.res_proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def _knn_incidence(self, pooled: torch.Tensor) -> torch.Tensor:
        B, N, C = pooled.shape
        dist = torch.cdist(pooled, pooled, p=2).pow(2)
        _, idx = dist.topk(min(self.knn_k + 1, N), dim=-1, largest=False)
        H = torch.zeros(B, N, N, device=pooled.device, dtype=pooled.dtype)
        H.scatter_(2, idx, 1.0)
        return H

    def _semantic_incidence(self, pooled: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("bnc,pc->bnp", pooled, self.proto) / math.sqrt(pooled.shape[-1])
        return F.softmax(scores, dim=-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, T, N, C = X.shape
        low = self.low_proj(X)
        high = self.high_proj(X)
        f_gcn = self.gcn(low)

        pooled = high.mean(dim=1)
        H_knn = self._knn_incidence(pooled)
        H_sem = self._semantic_incidence(pooled)
        H = torch.cat([H_sem, H_knn], dim=2)
        H = H[:, None].expand(B, T, N, self.num_proto + N).reshape(B * T, N, self.num_proto + N)
        f_hyp = self.hyper(high, H)

        gate = self.gate(torch.cat([f_gcn, f_hyp], dim=-1))
        fused = gate * f_gcn + (1.0 - gate) * f_hyp
        out = self.out_proj(fused)
        return self.norm(out + self.res_proj(X))


class MetaPartHypergraphFusionLite(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        H = torch.tensor([
            [1, 1, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
        ], dtype=torch.float32)
        self.register_buffer("H_meta", H)
        self.part_embed = nn.Parameter(torch.zeros(1, 1, 4, in_dim))
        nn.init.trunc_normal_(self.part_embed, std=0.02)
        self.hyper1 = HypergraphConv(in_dim, in_dim, num_edges=3, dropout=dropout)
        self.hyper2 = HypergraphConv(in_dim, in_dim, num_edges=3, dropout=dropout)
        self.out = nn.Sequential(
            nn.Linear(in_dim * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

    def _mean_part(self, X: torch.Tensor, start: int, end: int) -> torch.Tensor:
        return X[:, :, start:end, :].mean(dim=2)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        whole = X.mean(dim=2)
        left = self._mean_part(X, 0, 21)
        right = self._mean_part(X, 21, 42)
        body = self._mean_part(X, 42, X.shape[2])
        parts = torch.stack([whole, left, right, body], dim=2)
        z = parts + self.part_embed
        z = self.hyper1(z, self.H_meta)
        z = self.hyper2(z, self.H_meta)
        z = z + parts
        return self.out(z.reshape(z.shape[0], z.shape[1], -1))


class PhysicalBranch(nn.Module):
    def __init__(self, num_nodes: int, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        edges = _build_physical_edges(num_nodes)
        H     = _incidence_matrix(edges, num_nodes)   # (N, E)
        self.register_buffer('H', H)
        self.conv = HypergraphConv(in_dim, out_dim, H.shape[1], dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.conv(X, self.H)


class KNNBranch(nn.Module):
  
    def __init__(
        self,
        num_nodes: int,
        in_dim:    int,
        out_dim:   int,
        k:         int   = 5,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.k        = k
        self.num_nodes = num_nodes
        self.conv = HypergraphConv(in_dim, out_dim, num_nodes, dropout)

    def _build_knn_H(self, coords: torch.Tensor) -> torch.Tensor:

        BT, N, _ = coords.shape
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  
        dist = diff.pow(2).sum(-1)                         
        _, idx = dist.topk(min(self.k + 1, N), dim=-1, largest=False)  
        H = torch.zeros(BT, N, N, device=coords.device)
        H.scatter_(2, idx, 1.0)     
        return H   

    def forward(self, X: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, T, N, C = X.shape
        X_flat = X.reshape(B * T, N, C)
        c_flat = coords.reshape(B * T, N, 3)

        H = self._build_knn_H(c_flat)   

        H_mean = H.mean(0)              
        out = self.conv(X, H_mean)      
        return out

class SemanticBranch(nn.Module):


    def __init__(
        self,
        num_nodes:  int,
        in_dim:     int,
        out_dim:    int,
        num_proto:  int  = 16,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_proto = num_proto

        self.joint_key = nn.Parameter(torch.randn(num_nodes, in_dim))
        self.proto_key = nn.Parameter(torch.randn(num_proto, in_dim))
        nn.init.trunc_normal_(self.joint_key, std=0.02)
        nn.init.trunc_normal_(self.proto_key, std=0.02)

        self.conv = HypergraphConv(in_dim, out_dim, num_proto, dropout)
        self.scale = math.sqrt(in_dim)

    def _build_semantic_H(self) -> torch.Tensor:

        scores = self.joint_key @ self.proto_key.t() / self.scale
        return torch.sigmoid(scores)   

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        H = self._build_semantic_H()   # (N, num_proto)
        return self.conv(X, H)


class UncertaintyFusion(nn.Module):


    def __init__(self, num_branches: int, feat_dim: int):
        super().__init__()
        self.conf_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.GELU(),
                nn.Linear(feat_dim // 4, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_branches)
        ])

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
  
        confs = [head(f) for head, f in zip(self.conf_heads, feats)]  
        conf_sum = sum(confs)                          
        fused = sum(c * f for c, f in zip(confs, feats)) / (conf_sum + 1e-6)
        return fused  


class MultiScaleTemporalConv(nn.Module):
  
    def __init__(
        self,
        in_dim:   int,
        d_model:  int,
        kernels:  List[int] = [1, 3, 5, 7],
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_dim, in_dim,
                    kernel_size=k,
                    padding=k // 2,
                    groups=min(in_dim, 8),  
                    bias=False,
                ),
                nn.GELU(),
            )
            for k in kernels
        ])
        concat_dim = in_dim * len(kernels)
        self.proj = nn.Sequential(
            nn.Conv1d(concat_dim, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, in_dim) → (B, T, d_model)"""
        x = x.transpose(1, 2)                         
        branches = [b(x) for b in self.branches]
        x = torch.cat(branches, dim=1)                
        x = self.proj(x)                               
        x = x.transpose(1, 2)                         
        return self.norm(x)

class CTCHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.norm   = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(p=dropout)
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.norm(x))
        log_probs = F.log_softmax(self.linear(x), dim=-1)
        return log_probs.permute(1, 0, 2)


class Stream2_Hypergraph(nn.Module):
    def __init__(
        self,
        keypoint_dim: int        = 225,
        num_joints:   int        = 75,
        keypoint_layout: str     = "pose_hands",
        node_dim:     int        = 128,
        d_model:      int        = 512,
        num_proto:    int        = 16,
        knn_k:        int        = 5,
        ms_kernels:   List[int]  = [1, 3, 5, 7],
        dropout:      float      = 0.1,
        num_classes:  int        = 1296,
        ctc_dropout:  float      = 0.1,
        normalize_keypoints: bool = True,
        temporal_lstm_layers: int = 1,
        cgp_channels: List[int] = [64, 128, 128, 256],
    ):
        super().__init__()
        self.num_joints  = num_joints
        self.coord_dim   = keypoint_dim // num_joints   # 3
        self.keypoint_layout = keypoint_layout
        self.normalize_keypoints = normalize_keypoints
        first_dim = cgp_channels[0] if cgp_channels else node_dim
        self.joint_embed = nn.Sequential(
            nn.Linear(self.coord_dim, first_dim),
            nn.LayerNorm(node_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.joint_embed[1] = nn.LayerNorm(first_dim)

        cgp_blocks = []
        in_dim = first_dim
        for out_dim in cgp_channels:
            cgp_blocks.append(
                HyperSignLiteCGPBlock(
                    num_nodes=num_joints,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_proto=num_proto,
                    knn_k=knn_k,
                    dropout=dropout,
                )
            )
            in_dim = out_dim
        self.cgp_blocks = nn.ModuleList(cgp_blocks)
        self.part_fusion = MetaPartHypergraphFusionLite(in_dim=in_dim, out_dim=d_model, dropout=dropout)
        self.ms_temporal = MultiScaleTemporalConv(
            in_dim=d_model,
            d_model=d_model,
            kernels=ms_kernels,
            dropout=dropout,
        )
        self.temporal_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=temporal_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if temporal_lstm_layers > 1 else 0.0,
        ) if temporal_lstm_layers > 0 else None
        self.temporal_norm = nn.LayerNorm(d_model)

        self.ctc_head = CTCHead(d_model, num_classes, dropout=ctc_dropout)
        self.d_model  = d_model

    def _canonicalize_layout(self, coords: torch.Tensor) -> torch.Tensor:
        """Return hand-hand-pose order expected by the physical hyperedges."""
        if self.keypoint_layout in ("hands_pose", "right_left_pose"):
            return coords
        if self.keypoint_layout in ("pose_hands", "pose_right_left"):
            pose = coords[:, :, :33, :]
            hands = coords[:, :, 33:, :]
            return torch.cat([hands, pose], dim=2)
        raise ValueError(f"Unsupported keypoint_layout: {self.keypoint_layout}")

    def _normalize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Center and scale visible landmarks per frame while keeping missing nodes at zero."""
        valid = coords.abs().sum(dim=-1, keepdim=True) > 1e-6
        pose = coords[:, :, 42:, :]
        pose_valid = valid[:, :, 42:, :]

        shoulders = pose[:, :, [11, 12], :]
        shoulders_valid = pose_valid[:, :, [11, 12], :]
        shoulder_center = shoulders.mean(dim=2, keepdim=True)
        shoulder_ok = shoulders_valid.all(dim=2, keepdim=True)

        valid_count = valid.sum(dim=2, keepdim=True).clamp(min=1)
        fallback_center = (coords * valid.float()).sum(dim=2, keepdim=True) / valid_count
        center = torch.where(shoulder_ok, shoulder_center, fallback_center)

        shoulder_dist = (shoulders[:, :, 0, :] - shoulders[:, :, 1, :]).norm(dim=-1, keepdim=True)
        centered_abs = torch.where(valid, coords - center, torch.zeros_like(coords))

        hands = coords[:, :, :42, :]
        hands_valid = valid[:, :, :42, :]
        left_hand_relative = (hands[:, :, 0, :].abs().sum(dim=-1, keepdim=True) < 5e-2) & hands_valid[:, :, :21, :].any(dim=2)
        right_hand_relative = (hands[:, :, 21, :].abs().sum(dim=-1, keepdim=True) < 5e-2) & hands_valid[:, :, 21:, :].any(dim=2)
        hand_relative = torch.cat([
            left_hand_relative.unsqueeze(2).expand(-1, -1, 21, -1),
            right_hand_relative.unsqueeze(2).expand(-1, -1, 21, -1),
        ], dim=2)

        centered_hands = torch.where(hand_relative, hands, centered_abs[:, :, :42, :])
        centered = torch.cat([centered_hands, centered_abs[:, :, 42:, :]], dim=2)
        centered = torch.where(valid, centered, torch.zeros_like(centered))
        bbox_scale = centered_abs.abs().amax(dim=2).amax(dim=-1, keepdim=True).clamp(min=1e-3)
        scale = torch.where(shoulder_dist > 1e-3, shoulder_dist, bbox_scale).unsqueeze(-1)

        return torch.where(valid, centered / scale, torch.zeros_like(coords))

    def forward(
        self,
        keypoints:  torch.Tensor,   
        frame_lens: torch.Tensor, 
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, KD = keypoints.shape
        N = self.num_joints

        coords = keypoints.reshape(B, T, N, self.coord_dim) 
        coords = self._canonicalize_layout(coords)
        if self.normalize_keypoints:
            coords = self._normalize_coords(coords)

        X = self.joint_embed(coords)
        for block in self.cgp_blocks:
            X = block(X)

        feats = self.part_fusion(X)
        feats = self.ms_temporal(feats)
        if self.temporal_lstm is not None:
            lstm_out, _ = self.temporal_lstm(feats)
            feats = self.temporal_norm(feats + lstm_out)

        log_probs = self.ctc_head(feats)                
        out_lens  = frame_lens.clamp(max=T)

        return log_probs, feats, out_lens



if __name__ == "__main__":
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, T, KD  = 4, 64, 225
    num_classes = 1296

    model = Stream2_Hypergraph(
        keypoint_dim = KD,
        num_joints   = 75,
        node_dim     = 128,
        d_model      = 512,
        num_proto    = 16,
        knn_k        = 5,
        num_classes  = num_classes,
        dropout      = 0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Stream2 Hypergraph | params: {n_params:,}")

    kpts       = torch.randn(B, T, KD, device=device)
    frame_lens = torch.full((B,), T, dtype=torch.long, device=device)

    t0 = time.time()
    with torch.no_grad():
        log_probs, feats, out_lens = model(kpts, frame_lens)
    elapsed = time.time() - t0

    print(f"  Input keypoints: {list(kpts.shape)}")
    print(f"  log_probs      : {list(log_probs.shape)}  (T, B, num_classes)")
    print(f"  feats          : {list(feats.shape)}       (B, T, d_model)")
    print(f"  out_lens       : {out_lens.tolist()}")
    print(f"  Time           : {elapsed:.3f}s")
