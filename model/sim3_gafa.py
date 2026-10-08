import torch
import torch.nn as nn
import torch.nn.functional as F

from simeco.transformer_utils import index_points
from simeco.vec_layers import VecActivation, VecLinear


EPS = 1e-6


class Sim3InvariantLayer(nn.Module):
    def __init__(self, channels, out_dim=None, bias_epsilon=1e-6):
        super().__init__()
        out_dim = out_dim or channels
        self.fc_inv = VecLinear(channels, out_dim, mode="so3", bias_epsilon=0.0)
        self.fc_O = VecLinear(channels, out_dim, mode="sim3", bias_epsilon=bias_epsilon)
        self.fc_bias = VecLinear(channels, out_dim, mode="sim3", bias_epsilon=bias_epsilon)
        self.feature_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2), nn.LayerNorm(out_dim * 2), nn.GELU(),
            nn.Linear(out_dim * 2, out_dim), nn.LayerNorm(out_dim),
        )

    def forward(self, q, mem):
        global_feature = mem.mean(dim=1).unsqueeze(-1)
        bias = self.fc_bias(global_feature)
        z_so3 = self.fc_O(global_feature) - bias
        q = q - q.mean(dim=1, keepdim=True)
        v_inv_per_point = self.fc_inv(q.permute(0, 2, 3, 1).contiguous())
        v_inv_per_point = (v_inv_per_point * z_so3).sum(2).permute(0, 2, 1)
        v_inv_per_point = v_inv_per_point / (v_inv_per_point.norm(dim=-1, keepdim=True) + EPS)
        return self.feature_mlp(v_inv_per_point)


class Sim3GAFABlock(nn.Module):
    def __init__(self, channels, k, tau=0.2, bias_epsilon=1e-6):
        super().__init__()
        self.k, self.tau = k, tau
        act = nn.LeakyReLU(0.2, inplace=False)
        self.fc_in = nn.Sequential(
            VecLinear(channels, channels, mode="sim3", bias_epsilon=bias_epsilon),
            VecActivation(channels, act, mode="sim3", bias_epsilon=bias_epsilon),
            VecLinear(channels, channels, mode="sim3", bias_epsilon=bias_epsilon),
        )
        # Translation-free vector-type layers; bias-free SO(3) maps are scale homogeneous.
        self.fc_delta = nn.Sequential(
            VecLinear(1, 64, mode="so3", bias_epsilon=0.0),
            VecActivation(64, act, mode="so3", bias_epsilon=0.0),
            VecLinear(64, channels, mode="so3", bias_epsilon=0.0),
            VecActivation(channels, act, mode="so3", bias_epsilon=0.0),
            VecLinear(channels, channels, mode="so3", bias_epsilon=0.0),
        )
        self.fc_delta_1 = nn.Sequential(
            VecLinear(channels * 2, channels, mode="so3", bias_epsilon=0.0),
            VecActivation(channels, act, mode="so3", bias_epsilon=0.0),
            VecLinear(channels, channels, mode="so3", bias_epsilon=0.0),
        )
        self.fc_query = VecLinear(channels, channels, mode="so3", bias_epsilon=0.0)
        self.local_act = VecActivation(channels, act, mode="sim3", bias_epsilon=bias_epsilon)
        self.fc_delta_abs = nn.Sequential(
            VecLinear(2, 64, mode="sim3", bias_epsilon=bias_epsilon),
            VecActivation(64, act, mode="sim3", bias_epsilon=bias_epsilon),
            VecLinear(64, channels, mode="sim3", bias_epsilon=bias_epsilon),
        )
        self.fc_delta_l = nn.Sequential(
            VecLinear(1, 64, mode="so3", bias_epsilon=0.0),
            VecActivation(64, act, mode="so3", bias_epsilon=0.0),
            VecLinear(64, channels, mode="so3", bias_epsilon=0.0),
        )
        self.fuse_mlp = nn.Sequential(
            VecLinear(channels * 3, channels, mode="sim3", bias_epsilon=bias_epsilon),
            VecActivation(channels, act, mode="sim3", bias_epsilon=bias_epsilon),
            VecLinear(channels, channels, mode="sim3", bias_epsilon=bias_epsilon),
        )
        self.update_act = VecActivation(channels, act, mode="sim3", bias_epsilon=bias_epsilon)
        self.out_mlp = nn.Sequential(
            VecLinear(channels, channels, mode="sim3", bias_epsilon=bias_epsilon),
            VecActivation(channels, act, mode="sim3", bias_epsilon=bias_epsilon),
            VecLinear(channels, channels, mode="sim3", bias_epsilon=bias_epsilon),
        )
        self.out_act = VecActivation(channels, act, mode="sim3", bias_epsilon=bias_epsilon)

    def forward(self, query, query_pos, context, context_pos):
        with torch.no_grad():
            dis_mat = torch.cdist(query_pos, context_pos)
            knn_idx = dis_mat.topk(self.k, dim=-1, largest=False, sorted=False)[1]
        knn_xyz = index_points(context_pos, knn_idx)
        knn_feature = index_points(context, knn_idx)

        pre = query
        query = self.fc_in(query.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2)
        origin = query.mean(dim=2, keepdim=True)
        query_vec = query - origin
        knn_vec = knn_feature - origin.unsqueeze(2)

        pos_vec = query_pos.unsqueeze(2) - knn_xyz
        pos_enc = self.fc_delta(pos_vec.unsqueeze(3).permute(0, 3, 4, 1, 2).contiguous()).permute(0, 3, 4, 1, 2)
        knn_vec = self.fc_delta_1(torch.cat([knn_vec, pos_enc], dim=3)
                                  .permute(0, 3, 4, 1, 2).contiguous()).permute(0, 3, 4, 1, 2)
        query_key = self.fc_query(query_vec.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2)
        sim = F.softmax(F.cosine_similarity(query_key.flatten(2).unsqueeze(2),
                                            knn_vec.flatten(3), dim=-1) / self.tau, dim=2)
        update_vec = (sim[..., None, None] * knn_vec).sum(dim=2)
        query = self.local_act((pre + update_vec).permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2)

        pre = query
        center = query_pos.mean(dim=1, keepdim=True).expand_as(query_pos)
        pos_abs = torch.stack([query_pos, center], dim=2).permute(0, 2, 3, 1).contiguous()
        pos_enc_abs = self.fc_delta_abs(pos_abs).permute(0, 3, 1, 2)
        pair_pos = query_pos.unsqueeze(2) - query_pos.unsqueeze(1)
        pos_enc_l = self.fc_delta_l(pair_pos.unsqueeze(3).permute(0, 3, 4, 1, 2).contiguous())
        pos_enc_l = pos_enc_l.mean(dim=-1).permute(0, 3, 1, 2)
        pos_enc = pos_enc_abs + pos_enc_l

        query_global = query.mean(dim=1, keepdim=True).expand_as(query)
        new_point = self.fuse_mlp(torch.cat([query, query_global, pos_enc], dim=2)
                                  .permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2)
        update = new_point - new_point.mean(dim=2, keepdim=True)
        query = self.update_act((pre + update).permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2)

        pre = query
        new_point = self.out_mlp(query.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2)
        update = new_point - new_point.mean(dim=2, keepdim=True)
        return self.out_act((pre + update).permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2)


class Sim3GeometricAwareFeatureAggregator(nn.Module):
    def __init__(self, cfg, query_dim, context_dim, bias_epsilon=1e-6):
        super().__init__()
        dim = cfg.d_model
        self.query_proj = VecLinear(query_dim, dim, mode="sim3", bias_epsilon=bias_epsilon)
        self.context_proj = VecLinear(context_dim, dim, mode="sim3", bias_epsilon=bias_epsilon)
        self.inv = Sim3InvariantLayer(dim, bias_epsilon=bias_epsilon)
        self.blocks = nn.ModuleList([
            Sim3GAFABlock(dim, k, bias_epsilon=bias_epsilon) for k in cfg.K
        ])

    def forward(self, q, coarse_point_cloud, mem, coor):
        """
        q: [B, M, Cq, 3], coarse_point_cloud: [B, M, 3]
        mem: [B, N, Cm, 3], coor: [B, N, 3]
        """
        q = self.query_proj(q.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2)
        mem = self.context_proj(mem.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2)
        for block in self.blocks:
            q = block(q, coarse_point_cloud, mem, coor)
        invariant = self.inv(q, mem)
        return q, invariant


