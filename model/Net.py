import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.losses import ChamferDis, PoseDis, SmoothL1Dis, ChamferDis_wo_Batch
# from model.losses import ComPoseProxyLoss as Loss
from utils.data_utils import generate_augmentation
from model.modules import ModifiedResnet
from model.Net_modules import NOCS_Predictor, TwoStageSim3PoseEstimator
from model.sim3_gafa import Sim3GeometricAwareFeatureAggregator
from simeco.simeco_pipeline import PCTransformer, sim3Reconstructor


class Net(nn.Module):
    def __init__(self, cfg, simeco_cfg):
        super(Net, self).__init__()
        self.cat_num = cfg.cat_num
        self.cfg = cfg
        if cfg.rgb_backbone == "resnet":
            self.rgb_extractor = ModifiedResnet()
        elif cfg.rgb_backbone == 'dino':
            # frozen dino
            self.rgb_extractor = torch.hub.load('facebookresearch/dinov2','dinov2_vits14')
            for param in self.rgb_extractor.parameters():
                param.requires_grad = False

            self.feature_mlp = nn.Sequential(
                nn.Conv1d(384, 128, 1),
            )
        else:
            raise NotImplementedError
        
        # self.pts_extractor = PointNet2MSG(radii_list=[[0.01, 0.02], [0.02,0.04], [0.04,0.08], [0.08,0.16]])

        self.base_model = PCTransformer(simeco_cfg)
        
        # self.IAKD = InstanceAdaptiveKeypointDetector(cfg.IAKD)
        self.GAFA = Sim3GeometricAwareFeatureAggregator(
            cfg.GAFA,
            query_dim=simeco_cfg.decoder_config.embed_dim,
            context_dim=simeco_cfg.decoder_config.embed_dim,
            num_object_queries=simeco_cfg.num_query,
            num_proxy_queries=simeco_cfg.skeleton_num,
            bias_epsilon=simeco_cfg.bias_epsilon,
        )

        self.nocs_predictor = NOCS_Predictor(cfg.NOCS_Predictor)
        self.estimator = self.estimator = TwoStageSim3PoseEstimator(
                                            channels=cfg.GAFA.d_model,
                                            obj_mass=0.75,
                                            detach_nocs=True
                                        )

        self.reconstructor = sim3Reconstructor(cfg.Reconstructor)

        
        
    def forward(self, inputs):
        end_points = {}
        rgb = inputs['rgb']
        pts = inputs['pts']
        choose = inputs['choose']
        cls = inputs['category_label'].reshape(-1)
        num_obj = self.base_model.num_query
        b = pts.size(0)

        index = cls.long().to(pts.device) + torch.arange(
            b, device=pts.device, dtype=torch.long
        ) * self.cat_num

        # RGB feature extraction
        if self.cfg.rgb_backbone == 'resnet':
            rgb_local = self.rgb_extractor(rgb)
        elif self.cfg.rgb_backbone == 'dino':
            dino_feature = self.rgb_extractor.forward_features(rgb)['x_prenorm'][:, 1:]
            f_dim = dino_feature.shape[-1]
            num_patches = int(dino_feature.shape[1] ** 0.5)
            dino_feature = dino_feature.reshape(b, num_patches, num_patches, f_dim).permute(0, 3, 1, 2)
            dino_feature = F.interpolate(dino_feature, size=(num_patches * 14, num_patches * 14), mode='bilinear', align_corners=False)
            dino_feature = dino_feature.reshape(b, f_dim, -1)
            rgb_local = self.feature_mlp(dino_feature)
        else:
            raise NotImplementedError

        d = rgb_local.size(1)
        rgb_local = rgb_local.reshape(b, d, -1)
        choose = choose.unsqueeze(1).repeat(1, d, 1)
        rgb_local = torch.gather(rgb_local, 2, choose).contiguous()

        # Sim(3) data augmentation (training only)
        if self.training:
            delta_r, delta_t, delta_s = generate_augmentation(b)
            pts = (pts - delta_t) / delta_s.unsqueeze(2) @ delta_r

        # Object + Proxy query generation
        q, coarse_point_cloud, mem, coor, denoise_length = self.base_model(pts, rgb_local)
        clean_length = q.size(1) - denoise_length if denoise_length else q.size(1)
        q, kpt_3d = q[:, :clean_length], coarse_point_cloud[:, :clean_length]

        # Sim(3)-equivariant geometric feature aggregation
        q, q_inv = self.GAFA(q, kpt_3d, mem, coor)

        obj_q = q[:, :num_obj]
        obj_3d = kpt_3d[:, :num_obj]

        # Object reconstruction
        recon_model, recon_delta = self.reconstructor(obj_q, obj_3d)

        # Joint Object + Proxy NOCS prediction
        kpt_nocs = self.nocs_predictor(q_inv, index)

        # Two-stage Sim(3) pose estimation
        pose_outputs = self.estimator(
            kpt_3d, kpt_nocs, q, q_inv, num_obj,
            return_aux=self.training
        )

        if self.training:
            r, t, s, pose_aux = pose_outputs
        else:
            r, t, s = pose_outputs

        if self.training:
            # Restore keypoints to the original camera coordinate system
            end_points['pred_kpt_3d'] = (
                (kpt_3d @ delta_r.transpose(1, 2)) * delta_s.unsqueeze(2) + delta_t
            )

            # Restore reconstructed object points
            end_points['recon_model'] = (
                (recon_model.transpose(1, 2) @ delta_r.transpose(1, 2))
                * delta_s.unsqueeze(2) + delta_t
            )

            end_points['recon_delta'] = recon_delta
            end_points['pred_kpt_nocs'] = kpt_nocs

            # Final pose: reverse data augmentation
            end_points['pred_rotation'] = delta_r @ r
            end_points['pred_translation'] = (
                delta_t.squeeze(1)
                + delta_s * torch.bmm(delta_r, t.unsqueeze(2)).squeeze(2)
            )
            end_points['pred_size'] = s * delta_s

            # Stage 1 initial pose: reverse data augmentation
            end_points['pred_init_rotation'] = delta_r @ pose_aux['r0']
            end_points['pred_init_translation'] = (
                delta_t.squeeze(1)
                + delta_s * torch.bmm(delta_r, pose_aux['t0'].unsqueeze(2)).squeeze(2)
            )
            end_points['pred_init_scale'] = (
                pose_aux['scale0'] * delta_s.squeeze(1)
            )

        else:
            # No centering and no data augmentation during inference
            end_points['pred_translation'] = t
            end_points['pred_rotation'] = r
            end_points['pred_size'] = s
            end_points['pred_kpt_3d'] = kpt_3d
            end_points['kpt_nocs'] = kpt_nocs

        return end_points
