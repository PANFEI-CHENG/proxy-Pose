import torch
import torch.nn as nn
import torch.nn.functional as F

from rotation_utils import Ortho6d2Mat

from simeco.vec_layers import VecLinear

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

class Reconstructor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pts_per_kpt = cfg.pts_per_kpt
        self.ndim = cfg.ndim
        
        self.pos_enc = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, self.ndim, 1),
        )
        
        self.mlp = nn.Sequential(
            nn.Conv1d(self.ndim, self.ndim, 1),
            nn.ReLU(),
            nn.Conv1d(self.ndim, self.ndim, 1),
        )
        
        self.shape_decoder = nn.Sequential(
            nn.Conv1d(2*self.ndim, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3*self.pts_per_kpt, 1),
        )

    def forward(self, kpt_3d, kpt_feature):
        """
        Args:
            kpt_3d: (b, 3, kpt_num)
            kpt_feature: (b, c, kpt_num)

        Returns:
            recon_model: (b, 3, pts_per_kpt*kpt_num)
        """
        b = kpt_3d.shape[0]
        kpt_num = kpt_3d.shape[2]
        pos_enc_3d = self.pos_enc(kpt_3d) # (b, c, kpt_num)
        kpt_feature = self.mlp(kpt_feature) # (b, c, kpt_num)
        
        global_feature = torch.mean(pos_enc_3d + kpt_feature, dim=2, keepdim=True) # (b, c, 1)
        recon_feature = torch.cat([global_feature.repeat(1, 1, kpt_num), kpt_feature], dim=1) # (b, 2c, kpt_num)        
        # (b, 3*pts_per_kpt, kpt_num)
        recon_delta = self.shape_decoder(recon_feature)
        # (b, pts_per_kpt*kpt_num, 3)
        recon_delta = recon_delta.transpose(1, 2).reshape(b, kpt_num*self.pts_per_kpt, 3).contiguous()
        # (b, pts_per_kpt*kpt_num, 3)
        kpt_3d_interleave = kpt_3d.transpose(1, 2).repeat_interleave(self.pts_per_kpt, dim=1).contiguous()
        
        recon_model = (recon_delta + kpt_3d_interleave).transpose(1, 2).contiguous()
        return recon_model, recon_delta


class GeometricAwareFeatureAggregator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.block_num = cfg.block_num
        self.K = cfg.K
        self.d_model = cfg.d_model
        
        assert self.K.__len__() == self.block_num
        
        # build GAFA blocks
        self.GAFA_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.GAFA_blocks.append(GAFA_block(self.K[i], self.d_model))
        
    def forward(self, kpt_feature, kpt_3d, pts_feature, pts):
        """
        Args:
            kpt_feature: (b, kpt_num, dim)
            kpt_3d: (b, kpt_num, 3)
            pts_feature: (b, n, dim)
            pts: (b, n, 3)

        Returns:
            kpt_feature: (b, kpt_num, dim)
        """
        for i in range(self.block_num):
            kpt_feature = self.GAFA_blocks[i](kpt_feature, kpt_3d, pts_feature, pts)

        return kpt_feature


class GAFA_block(nn.Module):
    def __init__(self, k, d_model):
        super().__init__()
        self.k = k
        self.d_model = d_model
        
        self.fc_in = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
        )
        
        self.fc_delta = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.fc_delta_1 = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.fc_delta_l = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        
        self.fc_delta_abs = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        
        self.fuse_mlp = nn.Sequential(
            nn.Conv1d(3*d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
        
        self.out_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.relu = nn.ReLU()
        self.tau = 5.0
    
    def forward(self, kpt_feature, kpt_3d, pts_feature, pts):
        """_summary_

        Args:
            kpt_feature (_type_): (b, kpt_num, 2c)
            kpt_3d (_type_): (b, kpt_num, 3)
            pts_feature (_type_): (b, n, 2c)
            pts (_type_): (b, n, 3)

        Returns:
            kpt_feature: (b, kpt_num, 2c)
        """
        # (b, kpt_num, 1, 3) - (b, 1, n, 3) = (b, kpt_num, n, 3)     
        dis_mat = torch.norm(kpt_3d.unsqueeze(2) - pts.unsqueeze(1), dim=3)
        knn_idx = dis_mat.argsort()[:, :, :self.k]
        knn_xyz = index_points(pts, knn_idx)
        knn_feature = index_points(pts_feature, knn_idx)
        
        # (b, kpt_num, 2c)
        pre = kpt_feature 
        kpt_feature = self.fc_in(kpt_feature.transpose(1, 2))
        
        # (b, kpt_num, 1, 3) - (b, kpt_num, k, 3) = (b, kpt_num, k, 3)
        pos_enc = kpt_3d.unsqueeze(2) - knn_xyz
        pos_enc = self.fc_delta(pos_enc)
        
        # (b, kpt_num, k, 2c)
        knn_feature = self.fc_delta_1(torch.cat([knn_feature, pos_enc], dim=-1))
        
        sim = F.cosine_similarity(kpt_feature.transpose(1, 2).unsqueeze(2).repeat(1, 1, self.k, 1), knn_feature, dim=-1)
        sim = F.softmax(sim / self.tau, dim=2)
        # (b, kpt_num, 1, k) @ (b, kpt_num, k, 2c) -> (b, kpt_num, 1, 2c) -> (b, kpt_num, 2c)
        kpt_feature = torch.matmul(sim.unsqueeze(2), knn_feature).squeeze(2)
        kpt_feature = F.relu(kpt_feature + pre)
        
        # (b, kpt_num, 2c)
        pre = kpt_feature
        pos_enc_abs = self.fc_delta_abs(kpt_3d)
        
        # (b, kpt_num, 1, 3) - (b, 1, kpt_num, 3) = (b, kpt_num, kpt_num, 3)
        dis_mat_l = kpt_3d.unsqueeze(2) - kpt_3d.unsqueeze(1)
        pos_enc_abs_l = self.fc_delta_l(dis_mat_l)
        pos_enc_abs_l = torch.mean(pos_enc_abs_l, dim=2)
        
        # (b, kpt_num, 2c)
        pos_enc_l = pos_enc_abs + pos_enc_abs_l
        kpt_num = kpt_3d.shape[1]
        # (b, 1, 2c)
        kpt_global = torch.mean(kpt_feature, dim=1, keepdim=True)
        # (b, 4c, kpt_num) -> (b, 2c, kpt_num)
        kpt_feature = self.fuse_mlp(torch.cat([kpt_feature.transpose(1, 2), kpt_global.transpose(1, 2).repeat(1, 1, kpt_num), pos_enc_l.transpose(1, 2)], dim=1))
        # (b, kpt_num, 2c)
        kpt_feature = F.relu(kpt_feature.transpose(1, 2) + pre)
        pre = kpt_feature
        kpt_feature = self.out_mlp(kpt_feature)
        
        return self.relu(pre + kpt_feature)
    
class NOCS_Predictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bins_num = cfg.bins_num
        self.cat_num = cfg.cat_num
        bin_lenth = 1 / self.bins_num
        half_bin_lenth = bin_lenth / 2
        self.bins_center = torch.linspace(start=-0.5, end=0.5-bin_lenth, steps=self.bins_num).view(self.bins_num, 1).cuda()
        self.bins_center = self.bins_center + half_bin_lenth # (bins_num, 1)
        
        self.nocs_mlp = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
        )
        
        self.self_attn_layer = SelfAttnLayer(cfg.AttnLayer)
        
        self.atten_mlp = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, self.cat_num*3*self.bins_num, 1), 
        )
    
    def forward(self, kpt_feature, index):
        """_summary_

        Args:
            kpt_feature: (b, kpt_num, dim)
        Return:
            kpt_nocs:           (b, kpt_num, 3)
        """
        b, kpt_num, c = kpt_feature.shape
        kpt_feature = self.nocs_mlp(kpt_feature.transpose(1, 2)).transpose(1, 2) 
        kpt_feature = self.self_attn_layer(kpt_feature)
        # (b, self.cat_num*3*bins_num, kpt_num)
        attn = self.atten_mlp(kpt_feature.transpose(1, 2)) 
        attn = attn.view(b*self.cat_num, 3*self.bins_num, kpt_num).contiguous()
        attn = torch.index_select(attn, 0, index)
        attn = attn.view(b, 3, self.bins_num, kpt_num).contiguous()
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 3, 1, 2)
        kpt_nocs = torch.matmul(attn, self.bins_center).squeeze(-1)

        return kpt_nocs
        
class AttnBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=4, dim_ffn=256, dropout=0.0, dropout_attn=None):
        super(AttnBlock, self).__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.dropout3 = nn.Dropout(dropout, inplace=False)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_ffn),
                                 nn.ReLU(), nn.Dropout(dropout, inplace=False),
                                 nn.Linear(dim_ffn, d_model))
        
    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos
    
    
    def forward(self, kpt_query, input_feature):
        # cross-attn
        kpt_query2 = self.norm1(kpt_query)
        kpt_query2, attn = self.multihead_attn(query=kpt_query2,
                                   key=input_feature,
                                   value=input_feature
                                  )
        kpt_query = kpt_query + self.dropout1(kpt_query2)
        
        # self-attn
        kpt_query2 = self.norm2(kpt_query)
        kpt_query2, _ = self.self_attn(kpt_query2, 
                                               kpt_query2, 
                                               value=kpt_query2
                                                )
        kpt_query = kpt_query + self.dropout2(kpt_query2)
        
        # ffn
        kpt_query2 = self.norm3(kpt_query)
        kpt_query2 = self.ffn(kpt_query2)
        kpt_query = kpt_query + self.dropout3(kpt_query2)
        
        return kpt_query, attn
    
class AttnLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.block_num = cfg.block_num
        self.d_model = cfg.d_model
        self.num_head = cfg.num_head
        self.dim_ffn = cfg.dim_ffn
        
        # build attention blocks
        self.attn_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.attn_blocks.append(AttnBlock(d_model=self.d_model, num_heads=self.num_head, dim_ffn=self.dim_ffn, dropout=0.0, dropout_attn=None))

        
    def forward(self, batch_kpt_query, input_feature):
        """
            update kpt_query to instance-specific queries
        Args:
            batch_kpt_query: b, kpt_num, dim
            input_feature: b, dim, n

        Returns:
            batch_kpt_query: b, kpt_num, dim
            attn:  b, kpt_num, n
        """    
        input_feature = input_feature.transpose(1, 2) # (b, n, 2c)
        
        # (b, kpt_num, c)  (b, kpt_num, n)
        for i in range(self.block_num):
            batch_kpt_query, attn = self.attn_blocks[i](batch_kpt_query, input_feature)
        
        return batch_kpt_query, attn       
    
class SelfAttnBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=4, dim_ffn=256, dropout=0.0, dropout_attn=None):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
            
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_ffn),
                                 nn.ReLU(), nn.Dropout(dropout, inplace=False),
                                 nn.Linear(dim_ffn, d_model))
        
    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos
    
    
    def forward(self, kpt_query):
        
        # self-attn
        kpt_query2 = self.norm1(kpt_query)
        kpt_query2, _ = self.self_attn(kpt_query2, 
                                               kpt_query2, 
                                               value=kpt_query2
                                                )
        kpt_query = kpt_query + self.dropout1(kpt_query2)
        
        # ffn
        kpt_query2 = self.norm2(kpt_query)
        kpt_query2 = self.ffn(kpt_query2)
        kpt_query = kpt_query + self.dropout2(kpt_query2)
        
        return kpt_query
    
class SelfAttnLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.block_num = cfg.block_num
        self.d_model = cfg.d_model
        self.num_head = cfg.num_head
        self.dim_ffn = cfg.dim_ffn
        
        # build attention blocks
        self.attn_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.attn_blocks.append(SelfAttnBlock(d_model=self.d_model, num_heads=self.num_head, dim_ffn=self.dim_ffn, dropout=0.0, dropout_attn=None))

        
    def forward(self, batch_kpt_query):
        """
        Args:
            batch_kpt_query: b, kpt_num, dim
            
        Returns:
            batch_kpt_query: b, kpt_num, dim
        """    
        # (b, kpt_num, c)  (b, kpt_num, n)
        for i in range(self.block_num):
            batch_kpt_query = self.attn_blocks[i](batch_kpt_query)
        
        return batch_kpt_query    
    
class InstanceAdaptiveKeypointDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.kpt_num = cfg.kpt_num
        self.query_dim = cfg.query_dim
        # initialize shared kpt_query for all categories
        self.kpt_query = nn.Parameter(torch.empty(self.kpt_num, self.query_dim)) # (kpt_num, query_dim)
        nn.init.xavier_normal_(self.kpt_query)
        
        # build attention layer
        self.attn_layer = AttnLayer(cfg.AttnLayer)
        
    def forward(self, rgb_local, pts_local):
        """_summary_

        Args:
            rgb_local (_type_): (b, c, n)
            pts_local (_type_): (b, c, n)
            cls:                (b, )
        """
        b, c, n = rgb_local.shape
        
        input_feature = torch.cat((pts_local, rgb_local), dim=1)  # (b, 2c, n)
        
        batch_kpt_query = self.kpt_query.unsqueeze(0).repeat(b, 1, 1)
        # (b, kpt_num, 2c)  (b, kpt_num, n)
        batch_kpt_query, attn = self.attn_layer(batch_kpt_query, input_feature) 
        
        # cos similarity <a, b> / |a|*|b|
        norm1 = torch.norm(batch_kpt_query, p=2, dim=2, keepdim=True) 
        norm2 = torch.norm(input_feature, p=2, dim=1, keepdim=True) 
        heatmap = torch.bmm(batch_kpt_query, input_feature) / (norm1 * norm2 + 1e-7)
        heatmap = F.softmax(heatmap / 0.1, dim=2)
        
        return batch_kpt_query, heatmap
        
class PoseSizeEstimator(nn.Module):
    def __init__(self):
        super(PoseSizeEstimator, self).__init__()
        
        self.pts_mlp1 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.pts_mlp2 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.pose_mlp1 = nn.Sequential(
            nn.Conv1d(64+64+256, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
        )
        self.pose_mlp2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.rotation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )
        self.translation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.size_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        
    def forward(self, pts1, pts2, pts1_local):
        pts1 = self.pts_mlp1(pts1.transpose(1,2))
        pts2 = self.pts_mlp2(pts2.transpose(1,2))
        pose_feat = torch.cat([pts1, pts1_local.transpose(1,2), pts2], dim=1) # b, c, n

        pose_feat = self.pose_mlp1(pose_feat) # b, c, n
        pose_global = torch.mean(pose_feat, 2, keepdim=True) # b, c, 1
        pose_feat = torch.cat([pose_feat, pose_global.expand_as(pose_feat)], 1) # b, 2c, n
        pose_feat = self.pose_mlp2(pose_feat).squeeze(2) # b, c

        r = self.rotation_estimator(pose_feat)
        r = Ortho6d2Mat(r[:, :3].contiguous(), r[:, 3:].contiguous()).view(-1,3,3)
        t = self.translation_estimator(pose_feat)
        s = self.size_estimator(pose_feat)
        return r,t,s


class EquivariantPoseInitializer(nn.Module):
    def __init__(self, channels=256, min_scale=1e-5):
        super().__init__()
        self.channels = channels
        self.min_scale = min_scale

        # Four equivariant vectors: x-axis, y-axis, scale, translation offset
        self.vec_head = VecLinear(2 * channels, 4, mode="so3", bias_epsilon=0.0)

    @staticmethod
    def _orthogonalize(x, y):
        ex = F.normalize(x, dim=-1, eps=1e-7)
        ey = y - (ex * y).sum(dim=-1, keepdim=True) * ex
        ey = F.normalize(ey, dim=-1, eps=1e-7)
        ez = torch.cross(ex, ey, dim=-1)
        return torch.stack([ex, ey, ez], dim=-1)

    def forward(self, q, xyz, num_obj):
        b, n, c, d = q.shape
        if d != 3 or c != self.channels or xyz.shape != (b, n, 3):
            raise ValueError(f"Unexpected shape: q={tuple(q.shape)}, xyz={tuple(xyz.shape)}")
        if not 0 < num_obj < n:
            raise ValueError("num_obj must leave at least one Proxy Query")

        # Remove translation while preserving rotation and scale equivariance
        v = q - q.mean(dim=2, keepdim=True)

        # Aggregate Object and Proxy equivariant features
        obj_pool = v[:, :num_obj].mean(dim=1)       # [B,C,3]
        proxy_pool = v[:, num_obj:].mean(dim=1)     # [B,C,3]
        pooled = torch.cat([obj_pool, proxy_pool], dim=1)  # [B,2C,3]

        # Predict equivariant vectors
        vectors = self.vec_head(pooled)             # [B,4,3]

        # Initial rotation
        r0 = self._orthogonalize(vectors[:, 0], vectors[:, 1])

        # Initial scale from equivariant vector magnitude
        scale0 = vectors[:, 2].norm(dim=-1).clamp_min(self.min_scale)

        # Joint geometric center of Object and Proxy points
        joint_center = xyz.mean(dim=1)              # [B,3]

        # Initial translation
        t0 = joint_center + vectors[:, 3]           # [B,3]

        return r0, t0, scale0



class InvariantGuidedPoseRefiner(nn.Module):
    def __init__(self, channels=256, obj_mass=0.75, min_scale=1e-5):
        super().__init__()
        if not 0.0 < obj_mass < 1.0:
            raise ValueError("obj_mass must be strictly between 0 and 1")
        self.channels = channels
        self.obj_mass = obj_mass
        self.min_scale = min_scale

        # q_inv, initial-frame xyz, nocs, residual, ||residual||, type marker.
        self.score_head = nn.Sequential(
            nn.Linear(channels + 11, 128), nn.ReLU(inplace=False),
            nn.Linear(128, 64), nn.ReLU(inplace=False), nn.Linear(64, 1)
        )
        # Learn how much to trust R0, scale0, t0: three dimensionless strengths.
        self.prior_head = nn.Sequential(
            nn.Linear(2 * channels + 1, 128), nn.ReLU(inplace=False), nn.Linear(128, 3)
        )
        # Regress the *ratio* sx:sy:sz only. Overall scale comes from geometry.
        self.ratio_head = nn.Sequential(
            nn.Linear(2 * channels, 128), nn.ReLU(inplace=False), nn.Linear(128, 3)
        )
        # Initialize uniform within-branch correspondences and mild priors.
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)
        nn.init.zeros_(self.prior_head[-1].weight)
        nn.init.constant_(self.prior_head[-1].bias, -2.2)

    def forward(self, xyz, nocs, q_inv, r0, t0, scale0, num_obj, detach_nocs=True):
        b, n, d = xyz.shape
        if d != 3 or nocs.shape != xyz.shape or q_inv.shape != (b, n, self.channels):
            raise ValueError("xyz/nocs/q_inv shapes do not agree")
        if r0.shape != (b, 3, 3) or t0.shape != (b, 3) or scale0.shape != (b,):
            raise ValueError("Initial pose has unexpected shape")
        if not 0 < num_obj < n:
            raise ValueError("Must have both Object and Proxy keypoints")

        # At the beginning, let NOCS be trained by its own supervision.
        z = nocs.detach() if detach_nocs else nocs

        # Convert camera/local points into the initial canonical reference frame.
        # Row-vector convention: y = (x - t0) @ R0 / scale0.
        y = torch.bmm(xyz - t0[:, None], r0) / scale0[:, None, None].clamp_min(self.min_scale)
        residual = y - z
        residual_norm = residual.norm(dim=-1, keepdim=True)
        obj_type = torch.zeros_like(residual_norm)
        obj_type[:, num_obj:] = 1.0
        score_in = torch.cat([q_inv, y, z, residual, residual_norm, obj_type], dim=-1)
        logits = self.score_head(score_in).squeeze(-1)

        # Fixed total mass per branch prevents the more numerous points from dominating.
        w_obj = logits[:, :num_obj].softmax(dim=1) * self.obj_mass
        w_proxy = logits[:, num_obj:].softmax(dim=1) * (1.0 - self.obj_mass)
        weights = torch.cat([w_obj, w_proxy], dim=1)          # [B,N]; sums to 1

        obj_inv = q_inv[:, :num_obj].mean(dim=1)
        proxy_inv = q_inv[:, num_obj:].mean(dim=1)
        global_inv = torch.cat([obj_inv, proxy_inv], dim=-1)  # [B,2C]
        global_res = (weights * residual_norm.squeeze(-1)).sum(dim=1, keepdim=True)
        strengths = torch.sigmoid(self.prior_head(torch.cat([global_inv, global_res], dim=-1)))
        prior_rot, prior_scale, prior_trans = strengths.unbind(dim=-1)

        # Weighted centroids, all expressed in the initialization reference frame.
        mu_y = (weights[..., None] * y).sum(dim=1)  # [B,3]
        mu_z = (weights[..., None] * z).sum(dim=1)
        yc, zc = y - mu_y[:, None], z - mu_z[:, None]

        # H = sum w * (y - mean_y) (z - mean_z)^T.
        H = torch.bmm((weights[..., None] * yc).transpose(1, 2), zc)  # [B,3,3]
        var_z = (weights * zc.square().sum(dim=-1)).sum(dim=1)        # [B]

        # An identity prior on the *relative* rotation preserves first-stage influence.
        eye = torch.eye(3, dtype=H.dtype, device=H.device).expand(b, 3, 3)
        # Small diagonal anisotropic jitter avoids identical singular values at zero H.
        jitter = H.new_tensor([1e-5, 2e-5, 3e-5]).view(1, 3)
        H_prior = H + (prior_rot * var_z + 1e-5)[:, None, None] * eye
        H_prior = H_prior + torch.diag_embed(jitter.expand(b, -1))

        # SVD and reflection handling; fit y ~= gamma * R_delta * z + delta.
        U, _, Vh = torch.linalg.svd(H_prior, full_matrices=False)
        uv = U @ Vh
        handedness = (torch.cross(uv[:, 0], uv[:, 1], dim=-1) * uv[:, 2]).sum(dim=-1)
        reflection = torch.where(handedness < 0, -torch.ones_like(handedness), torch.ones_like(handedness))
        d_fix = torch.stack([torch.ones_like(reflection), torch.ones_like(reflection), reflection], dim=-1)
        R_delta = U @ torch.diag_embed(d_fix) @ Vh

        # Scalar scale anchored to gamma=1: no unrestricted scale explosion for bad NOCS.
        scale_prior = prior_scale * var_z + 1e-5
        gamma_num = (R_delta * H).sum(dim=(-2, -1)) + scale_prior
        gamma = (gamma_num / (var_z + scale_prior)).clamp(min=0.25, max=4.0)

        # Translation anchored to delta=0, i.e. to initial t0.
        mu_z_rot = torch.bmm(mu_z[:, None], R_delta.transpose(1, 2)).squeeze(1)
        delta = (mu_y - gamma[:, None] * mu_z_rot) / (1.0 + prior_trans[:, None])

        # Compose relative local-frame correction with the equivariant initializer.
        r = torch.bmm(r0, R_delta)
        scale = scale0 * gamma
        t = t0 + scale0[:, None] * torch.bmm(delta[:, None, :], r0.transpose(1, 2)).squeeze(1)

        # Three-axis size vector with norm equal to geometric similarity scale.
        ratio = F.softplus(self.ratio_head(global_inv)) + 1e-5
        ratio = F.normalize(ratio, p=2, dim=-1, eps=1e-8)
        size = scale[:, None] * ratio
        info = {"weights": weights, "prior_strength": strengths, "relative_scale": gamma}
        return r, t, size, info


class TwoStageSim3PoseEstimator(nn.Module):
    def __init__(self, channels=256, obj_mass=0.75, detach_nocs=True):
        super().__init__()
        self.initializer = EquivariantPoseInitializer(channels=channels)
        self.refiner = InvariantGuidedPoseRefiner(channels=channels, obj_mass=obj_mass)
        self.detach_nocs = detach_nocs

    def forward(self, xyz, nocs, q, q_inv, num_obj, return_aux=False):
        """Outputs r,t,size; optionally also stage-one predictions for auxiliary loss."""
        # Keep Gram-Schmidt and differentiable SVD in FP32 under AMP.
        with torch.amp.autocast(device_type=xyz.device.type, enabled=False):
            r0, t0, scale0 = self.initializer(q.float(), xyz.float(), num_obj)
            r, t, size, info = self.refiner(
                xyz.float(), nocs.float(), q_inv.float(),
                r0.float(), t0.float(), scale0.float(),
                num_obj, detach_nocs=self.detach_nocs,
            )
        if not return_aux:
            return r, t, size
        info.update({"r0": r0, "t0": t0, "scale0": scale0})
        return r, t, size, info
