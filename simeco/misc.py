import torch

try:
    from pointnet2_ops import pointnet2_utils
except ImportError:
    import pointnet2_utils


def fps(points, number):
    """Sample ``number`` points with farthest-point sampling."""
    indices = pointnet2_utils.furthest_point_sample(points, number)
    return pointnet2_utils.gather_operation(
        points.transpose(1, 2).contiguous(), indices
    ).transpose(1, 2).contiguous()


def jitter_points(points, sigma=0.01, clip=0.05):
    """Apply clipped Gaussian jitter used by SIMECO's denoising queries."""
    noise = torch.randn_like(points).mul_(sigma).clamp_(-clip, clip)
    return points + noise
