import torch
import torch.nn as nn
from functools import partial

try:
    from timm.models.layers import trunc_normal_
except ImportError:
    from torch.nn.init import trunc_normal_
try:
    from pointnet2_ops import pointnet2_utils
except ImportError:
    # AG-Pose already ships the legacy PointNet2 extension under this name.
    import pointnet2_utils

from . import misc

from . import MODELS
from .vec_layers import *
from .transformer_utils import *



def build_semantic_self_mask(num_obj, num_proxy, num_dn, device):
    o, p, d = num_obj, num_proxy, num_dn
    mask = torch.ones(o + p + d, o + p + d, dtype=torch.bool, device=device)

    mask[:o, :o] = False
    mask[o:o+p, :o+p] = False

    if d > 0:
        mask[o+p:, :o] = False
        mask[o+p:, o+p:] = False

    return mask


def build_semantic_self_knn(pos, num_obj, num_proxy, num_dn, k, proxy_obj_k=4):
    o, p, d = num_obj, num_proxy, num_dn
    proxy_k = k - proxy_obj_k

    if pos.size(1) != o + p + d:
        raise ValueError("Query count does not match Object/Proxy/Denoising sizes")
    if not (0 < proxy_obj_k < k) or o < k or p < proxy_k:
        raise ValueError("Invalid KNN configuration")

    with torch.no_grad():
        obj_pos = pos[:, :o]
        proxy_pos = pos[:, o:o+p]

        obj_idx = knn_point(k, obj_pos, obj_pos)

        proxy_obj_idx = knn_point(proxy_obj_k, obj_pos, proxy_pos)
        proxy_proxy_idx = knn_point(proxy_k, proxy_pos, proxy_pos) + o
        proxy_idx = torch.cat([proxy_obj_idx, proxy_proxy_idx], dim=-1)

        if d > 0:
            dn_pos = pos[:, o+p:]
            dn_context = torch.cat([obj_pos, dn_pos], dim=1)
            dn_idx = knn_point(k, dn_context, dn_pos)
            dn_idx = dn_idx + (dn_idx >= o).long() * p
            return torch.cat([obj_idx, proxy_idx, dn_idx], dim=1)

        return torch.cat([obj_idx, proxy_idx], dim=1)


class VecSelfAttnBlockApi(nn.Module):
    """
    Vector-based Self-Attention Block for Equivariant Networks.
    
    This module implements a self-attention mechanism that operates on vector features
    while maintaining equivariance. It supports both global attention and local
    graph-based attention mechanisms.
    """
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        mlp_ratio: float = 4.0, 
        init_values=None,
        scale_factor=None,
        act_layer=None,
        block_style: str = "vnattn-vngraph",
        combine_style: str = "concat",
        k: int = 10, 
        mode: str = "so3",
        bias_epsilon: float = 1e-6
    ):
        """
        Initialize VecSelfAttnBlockApi.

        Args:
            dim (int): Feature dimension
            num_heads (int): Number of attention heads
            mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension
            init_values: Initial values for layer scaling
            scale_factor: Scaling factor for feature transformation
            act_layer: Activation layer for MLP
            block_style (str): Style of attention blocks ('vnattn', 'vngraph', or combination)
            combine_style (str): How to combine multiple attention outputs ('concat' only)
            k (int): Number of neighbors for local graph attention
            mode (str): Equivariance mode ('so3' for rotation equivariance)
            bias_epsilon (float): Small constant for numerical stability
        """
        super().__init__()

        # Validate combine style - only concat is supported
        if combine_style != "concat":
            raise ValueError(f"Unsupported combine_style: {combine_style}, only 'concat' is allowed.")

        # Initialize normalization layers for translation invariance
        self.norm1 = VecLayerNorm(dim, eps=bias_epsilon)  # Pre-attention normalization
        self.norm2 = VecLayerNorm(dim, eps=bias_epsilon)  # Pre-MLP normalization
        
        # Layer scaling for training stability (optional)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        
        # Store configuration
        self.mode = mode
        self.scale_factor = scale_factor

        # Set default activation if not provided
        if act_layer is None:
            act_layer = nn.LeakyReLU(negative_slope=0.2) 

        # Multi-layer perceptron for feature transformation
        # Operates on vector features while maintaining equivariance
        self.mlp = nn.Sequential(
            VecLinear(v_in=dim, v_out=int(dim * mlp_ratio), mode="so3", bias_epsilon=bias_epsilon),
            VecActivation(int(dim * mlp_ratio), act_func=act_layer, mode="so3", bias_epsilon=bias_epsilon),
            VecLinear(v_in=int(dim * mlp_ratio), v_out=dim, mode="so3", bias_epsilon=bias_epsilon),
        )      

        # Parse block style to determine which attention mechanisms to use
        block_tokens = block_style.split('-')
        if not (0 < len(block_tokens) <= 2):
            raise ValueError(f"Invalid block_style: {block_style}.")

        # Initialize attention mechanisms based on block style
        self.attn = None        # Global vector attention
        self.local_attn = None  # Local graph-based attention

        for block_token in block_tokens:
            if block_token == "vnattn":
                # Vector neuron attention for global relationships
                self.attn = VNAttention(dim, num_heads=num_heads, mode="so3", bias_epsilon=bias_epsilon)
            elif block_token == "vngraph":
                # Dynamic graph attention for local neighborhoods
                self.local_attn = VNDynamicGraphAttention(dim, k=k, bias_epsilon=bias_epsilon)
            else:
                raise ValueError(f"Unexpected block_token: {block_token}. Supported: 'vnattn', 'vngraph'.")

        self.block_length = len(block_tokens)

        # Create merge layer if both attention types are present
        if self.attn and self.local_attn:
            self.merge_map = VecLinear(v_in=dim * 2, v_out=dim, mode="so3", bias_epsilon=bias_epsilon)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, idx: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of VecSelfAttnBlockApi.

        Args:
            x (torch.Tensor): Input feature tensor of shape [B, C, 3, N]
                             where B=batch, C=channels, 3=vector dimension, N=points
            pos (torch.Tensor): Position tensor of shape [B, N, 3] for spatial relationships
            idx (torch.Tensor, optional): Neighbor indices for local attention [B, N, k]

        Returns:
            torch.Tensor: Output tensor of same shape as input after self-attention and MLP
        """
        attn_features = []

        # Apply layer normalization for translation invariance
        norm_x = self.norm1(x)

        # Compute attention features based on available attention mechanisms
        if self.attn:
            # Global vector attention
            attn_features.append(self.attn(norm_x))
        if self.local_attn:
            # Local graph attention using spatial neighborhoods
            attn_features.append(self.local_attn(norm_x, pos, idx=idx))

        # Combine attention outputs and apply residual connection
        if len(attn_features) == 1:
            # Single attention mechanism - direct residual connection
            x = x + self.ls1(transform_restore(x, attn_features[0], self.scale_factor))
        elif len(attn_features) == 2:
            # Multiple attention mechanisms - concatenate and merge
            merged_features = torch.cat(attn_features, dim=-2)  # Concatenate along channel dimension
            # Apply linear transformation to reduce back to original dimension
            merged_features = self.merge_map(merged_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = x + self.ls1(transform_restore(x, merged_features, self.scale_factor))
        else:
            raise RuntimeError(f"Unexpected number of attention features: {len(attn_features)}")

        # Apply MLP with residual connection
        # Permute for vector operations, then restore original format
        mlp_input = self.norm2(x).permute(0, 2, 3, 1)  # [B, 3, N, C]
        mlp_output = self.mlp(mlp_input).permute(0, 3, 1, 2)  # Back to [B, C, 3, N]
        x = x + self.ls2(transform_restore(x, mlp_output, self.scale_factor))
        
        return x


class VecCrossAttnBlockApi(nn.Module):
    """
    Vector-based Cross-Attention Block for Equivariant Networks.
    
    This module implements both self-attention and cross-attention mechanisms
    that operate on vector features while maintaining equivariance. It supports
    global attention and local graph-based attention for both self and cross
    attention operations.
    """
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        mlp_ratio: float = 4.0, 
        init_values=None,
        scale_factor=None,
        act_layer=None,  
        self_attn_block_style: str = "attn-deform", 
        self_attn_combine_style: str = "concat",
        cross_attn_block_style: str = "attn-deform", 
        cross_attn_combine_style: str = "concat",
        k: int = 10, 
        mode: str = "so3",
        bias_epsilon: float = 1e-6
    ):
        """
        Initialize VecCrossAttnBlockApi.

        Args:
            dim (int): Feature dimension
            num_heads (int): Number of attention heads
            mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension
            init_values: Initial values for layer scaling
            scale_factor: Scaling factor for feature transformation
            act_layer: Activation layer for MLP
            self_attn_block_style (str): Style of self-attention blocks
            self_attn_combine_style (str): How to combine self-attention outputs
            cross_attn_block_style (str): Style of cross-attention blocks
            cross_attn_combine_style (str): How to combine cross-attention outputs
            k (int): Number of neighbors for local graph attention
            mode (str): Equivariance mode ('so3' for rotation equivariance)
            bias_epsilon (float): Small constant for numerical stability
        """
        super().__init__()        

        # Set default activation if not provided
        if act_layer is None:
            act_layer = nn.LeakyReLU(negative_slope=0.2)

        # Initialize normalization layers for translation invariance
        self.norm1 = VecLayerNorm(dim, eps=bias_epsilon)  # Pre-self-attention normalization
        self.norm2 = VecLayerNorm(dim, eps=bias_epsilon)  # Pre-MLP normalization
        self.norm_q = VecLayerNorm(dim, eps=bias_epsilon)  # Query normalization for cross-attention
        self.norm_v = VecLayerNorm(dim, eps=bias_epsilon)  # Value normalization for cross-attention
        
        # Store configuration parameters
        self.mode = mode
        self.scale_factor = scale_factor
        
        # Layer scaling for training stability (optional)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()

        # Multi-layer perceptron for feature transformation
        # Operates on vector features while maintaining equivariance
        self.mlp = nn.Sequential(
            VecLinear(v_in=dim, v_out=int(dim * mlp_ratio), mode="so3", bias_epsilon=bias_epsilon),
            VecActivation(int(dim * mlp_ratio), act_func=act_layer, mode="so3", bias_epsilon=bias_epsilon),
            VecLinear(v_in=int(dim * mlp_ratio), v_out=dim, mode="so3", bias_epsilon=bias_epsilon),
        )

        # Build self-attention mechanisms based on block style
        self.self_attn, self.local_self_attn, self.self_attn_merge_map = self.build_attention(
            block_style=self_attn_block_style, 
            combine_style=self_attn_combine_style, 
            dim=dim, 
            num_heads=num_heads, 
            k=k,
            bias_epsilon=bias_epsilon
        )

        # Build cross-attention mechanisms based on block style
        self.cross_attn, self.local_cross_attn, self.cross_attn_merge_map = self.build_attention(
            block_style=cross_attn_block_style, 
            combine_style=cross_attn_combine_style, 
            dim=dim, 
            num_heads=num_heads, 
            k=k,
            is_cross=True,  # Flag to indicate cross-attention
            bias_epsilon=bias_epsilon
        )

    def build_attention(self, block_style, combine_style, dim, num_heads, k, is_cross=False, bias_epsilon=1e-6):
        """
        Build attention mechanisms based on block style configuration.
        
        Args:
            block_style (str): Attention block style (e.g., 'vnattn-vngraph')
            combine_style (str): How to combine multiple attention outputs
            dim (int): Feature dimension
            num_heads (int): Number of attention heads
            k (int): Number of neighbors for graph attention
            is_cross (bool): Whether this is for cross-attention
            bias_epsilon (float): Small constant for numerical stability
            
        Returns:
            tuple: (global_attention, local_attention, merge_layer)
        """
        # Parse block style tokens
        block_tokens = block_style.split('-')
        if not (0 < len(block_tokens) <= 2):
            raise ValueError(f"Invalid block_style: {block_style}. Expected 'attn', 'deform', or both.")

        attn = None        # Global vector attention
        local_attn = None  # Local graph-based attention

        # Initialize attention mechanisms based on tokens
        for block_token in block_tokens:
            if block_token == "vnattn":
                # Vector neuron attention for global relationships
                attn = VNAttention(dim, num_heads=num_heads, mode="so3") if not is_cross else VNCrossAttention(dim, dim, num_heads=num_heads, mode="so3", bias_epsilon=bias_epsilon)
            elif block_token == "vngraph":
                # Dynamic graph attention for local neighborhoods
                local_attn = VNDynamicGraphAttention(dim, k=k, bias_epsilon=bias_epsilon) 
            else:
                raise ValueError(f"Unexpected block_token: {block_token}. Supported: 'vnattn', 'vngraph'.")

        # Create merge layer if both attention types are present
        merge_map = None
        if attn and local_attn and combine_style == "concat":
            merge_map = VecLinear(v_in=dim * 2, v_out=dim, mode="so3", bias_epsilon=bias_epsilon)

        return attn, local_attn, merge_map

    def apply_attention(self, x, pos, idx, attn, local_attn, merge_map, layer_scale, scale_factor=None, mask=None, is_cross=False, v=None, v_pos=None):
        """
        Apply attention mechanisms and combine their outputs.
        
        Args:
            x (torch.Tensor): Input query tensor
            pos (torch.Tensor): Position tensor for spatial relationships
            idx (torch.Tensor): Neighbor indices for local attention
            attn: Global attention mechanism
            local_attn: Local graph attention mechanism
            merge_map: Layer to merge multiple attention outputs
            layer_scale: Layer scaling module
            scale_factor: Scaling factor for features
            mask (torch.Tensor): Attention mask (for self-attention)
            is_cross (bool): Whether this is cross-attention
            v (torch.Tensor): Value tensor (for cross-attention)
            v_pos (torch.Tensor): Value positions (for cross-attention)
            
        Returns:
            torch.Tensor: Output tensor after applying attention and residual connection
        """
        attn_features = []
        
        # Apply normalization based on attention type
        norm_x = self.norm1(x) if not is_cross else self.norm_q(x)
        norm_v = self.norm_v(v) if is_cross else None

        # Compute attention features based on available mechanisms
        if attn:
            # Global attention (self or cross)
            attn_features.append(attn(norm_x, mask=mask) if not is_cross else attn(norm_x, norm_v))
        if local_attn:
            # Local graph attention (self or cross)
            attn_features.append(
                local_attn(norm_x, pos, idx=idx) if not is_cross else 
                local_attn(q=norm_x, v=norm_v, q_pos=pos, v_pos=v_pos, idx=idx)
            )

        # Combine attention outputs and apply residual connection
        if len(attn_features) == 1:
            # Single attention mechanism - direct residual connection
            return x + layer_scale(transform_restore(x, attn_features[0], scale_factor))
        elif len(attn_features) == 2 and merge_map:
            # Multiple attention mechanisms - concatenate and merge
            merged_features = torch.cat(attn_features, dim=-2)  # Concatenate along channel dimension
            merged_features = merge_map(merged_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            return x + layer_scale(transform_restore(x, merged_features, scale_factor))
        else:
            raise RuntimeError(f"Unexpected number of attention features: {len(attn_features)}")


    def forward(self, q, v, q_pos, v_pos, self_attn_idx=None,
                cross_attn_idx=None, denoise_length=None, self_attn_mask=None):

        mask = self_attn_mask
        if mask is None and denoise_length is not None:
            query_len = q.size(1)
            mask = torch.zeros(query_len, query_len, dtype=torch.bool, device=q.device)
            mask[:-denoise_length, -denoise_length:] = True

        # Self-Attention
        q = self.apply_attention(
            x=q, pos=q_pos, idx=self_attn_idx,
            attn=self.self_attn, local_attn=self.local_self_attn,
            merge_map=self.self_attn_merge_map, mask=mask,
            layer_scale=self.ls1, scale_factor=self.scale_factor
        )

        # Cross-Attention with shared memory
        q = self.apply_attention(
            x=q, v=v, pos=q_pos, v_pos=v_pos, idx=cross_attn_idx,
            attn=self.cross_attn, local_attn=self.local_cross_attn,
            merge_map=self.cross_attn_merge_map, is_cross=True,
            layer_scale=self.ls2, scale_factor=self.scale_factor
        )

        # MLP
        mlp_input = self.norm2(q).permute(0, 2, 3, 1)
        mlp_output = self.mlp(mlp_input).permute(0, 3, 1, 2)
        q = q + self.ls2(transform_restore(q, mlp_output, self.scale_factor))

        return q



class TransformerEncoder(nn.Module):
    """ 
    Transformer Encoder without hierarchical structure

    This encoder applies a sequence of VecSelfAttnBlockApi blocks.
    """
    
    def __init__(
        self, 
        embed_dim=256, 
        depth=4, 
        num_heads=4, 
        mlp_ratio=4.,  
        init_values=None,
        norm_layer=None,
        act_layer=nn.GELU(), 
        block_style_list=['vnattn-vngraph'], 
        combine_style='concat', 
        k=10, 
        mode="so3",
        bias_epsilon=1e-6,
        scale_factor=None
    ):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        
        for i in range(depth):
            self.blocks.append(VecSelfAttnBlockApi(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                init_values=init_values,
                act_layer=act_layer, 
                block_style=block_style_list[i], 
                combine_style=combine_style, 
                k=k, 
                mode=mode, 
                bias_epsilon=bias_epsilon, 
                scale_factor=scale_factor
            ))

    def forward(self, x, pos):
        idx = knn_point(self.k, pos, pos)
        for _, block in enumerate(self.blocks):
            x = block(x, pos, idx=idx) 
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder without hierarchical structure.
    
    This decoder applies a sequence of VecCrossAttnBlockApi blocks.
    """
    
    def __init__(
        self, 
        embed_dim=256, 
        depth=4, 
        num_heads=4, 
        mlp_ratio=4., 
        init_values=None,
        act_layer=None, 
        norm_layer=None,
        self_attn_block_style_list=['vnattn-vngraph'], 
        self_attn_combine_style='concat',
        cross_attn_block_style_list=['vnattn-vngraph'], 
        cross_attn_combine_style='concat',
        k=10,
        proxy_obj_k=4,
        mode="so3",
        bias_epsilon=1e-6, 
        scale_factor=None
    ):
        super().__init__()
        
        # Set default values
        if act_layer is None:
            act_layer = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            
        self.k = k
        self.proxy_obj_k = proxy_obj_k
        self.blocks = nn.ModuleList()
        
        # Build decoder blocks
        for i in range(depth):
            self.blocks.append(VecCrossAttnBlockApi(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                init_values=init_values,
                scale_factor=scale_factor,
                act_layer=act_layer,
                self_attn_block_style=self_attn_block_style_list[i], 
                self_attn_combine_style=self_attn_combine_style,
                cross_attn_block_style=cross_attn_block_style_list[i], 
                cross_attn_combine_style=cross_attn_combine_style,
                k=k, 
                mode=mode, 
                bias_epsilon=bias_epsilon
            ))


    def forward(self, q, v, q_pos, v_pos, denoise_length=None,
                num_object_queries=None, num_proxy_queries=None):

        if num_object_queries is not None and num_proxy_queries is not None:
            o = int(num_object_queries)
            p = int(num_proxy_queries)
            d = int(denoise_length or 0)

            self_attn_mask = build_semantic_self_mask(o, p, d, q_pos.device)
            self_attn_idx = build_semantic_self_knn(
                q_pos, o, p, d, self.k, self.proxy_obj_k
            )
        else:
            self_attn_mask = None
            self_attn_idx = knn_point(self.k, q_pos, q_pos) if denoise_length is None else None

        cross_attn_idx = knn_point(self.k, v_pos, q_pos)

        for block in self.blocks:
            q = block(
                q=q, v=v, q_pos=q_pos, v_pos=v_pos,
                self_attn_idx=self_attn_idx,
                cross_attn_idx=cross_attn_idx,
                denoise_length=denoise_length,
                self_attn_mask=self_attn_mask
            )

        return q


class PointTransformerEncoder(nn.Module):
    """
    Vision Transformer for point cloud encoder/decoder
    
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    
    def __init__(
        self, 
        embed_dim=256, 
        depth=12, 
        num_heads=4, 
        mlp_ratio=4., 
        init_values=None,
        norm_layer=None, 
        act_layer=None,
        block_style_list=['vnattn-vngraph'], 
        combine_style='concat',
        k=10, 
        mode="so3",
        bias_epsilon=1e-6, 
        scale_factor=None
    ):
        """
        Initialize PointTransformerEncoder.

        Args:
            embed_dim (int): Embedding dimension
            depth (int): Depth of transformer
            num_heads (int): Number of attention heads
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
            init_values: Layer-scale init values
            norm_layer: Normalization layer
            act_layer: MLP activation layer
            block_style_list (list): List of block styles for each layer
            combine_style (str): How to combine attention outputs
            k (int): Number of neighbors for graph attention
            mode (str): Equivariance mode
            bias_epsilon (float): Small constant for numerical stability
            scale_factor: Scaling factor for features
        """
        super().__init__()
        
        # Set default layers
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.LeakyReLU(negative_slope=0.2, inplace=False)
        
        # Store configuration
        self.num_features = self.embed_dim = embed_dim

        assert len(block_style_list) == depth
        
        # Build transformer blocks
        self.blocks = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            init_values=init_values,
            norm_layer=norm_layer, 
            act_layer=act_layer,
            block_style_list=block_style_list,
            combine_style=combine_style,
            k=k,
            mode=mode,
            bias_epsilon=bias_epsilon,
            scale_factor=scale_factor
        )

        self.norm = norm_layer(embed_dim) 
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for different layer types."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, VecLinear):
            trunc_normal_(m.weight, std=.02)

    def forward(self, x, pos):
        """
        Forward pass of PointTransformerEncoder.
        
        Args:
            x (torch.Tensor): Input feature tensor
            pos (torch.Tensor): Position tensor
            
        Returns:
            torch.Tensor: Encoded features
        """
        x = self.blocks(x, pos)
        return x


class PointTransformerDecoder(nn.Module):
    """
    Vision Transformer for point cloud encoder/decoder
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    
    def __init__(
        self, 
        embed_dim=256, 
        depth=12, 
        num_heads=4, 
        mlp_ratio=4., 
        init_values=None,
        norm_layer=None, 
        act_layer=None,
        self_attn_block_style_list=['vnattn-vngraph'], 
        self_attn_combine_style='concat',
        cross_attn_block_style_list=['vnattn-vngraph'], 
        cross_attn_combine_style='concat',
        k=10,
        proxy_obj_k=4,
        mode="so3",
        bias_epsilon=1e-6, 
        scale_factor=None
    ):
        """
        Initialize PointTransformerDecoder.

        Args:
            embed_dim (int): Embedding dimension
            depth (int): Depth of transformer
            num_heads (int): Number of attention heads
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
            init_values: Layer-scale init values
            norm_layer: Normalization layer
            act_layer: MLP activation layer
            self_attn_block_style_list (list): List of self-attention block styles
            self_attn_combine_style (str): How to combine self-attention outputs
            cross_attn_block_style_list (list): List of cross-attention block styles
            cross_attn_combine_style (str): How to combine cross-attention outputs
            k (int): Number of neighbors for graph attention
            mode (str): Equivariance mode
            bias_epsilon (float): Small constant for numerical stability
            scale_factor: Scaling factor for features
        """
        super().__init__()
        
        # Set default layers
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.LeakyReLU(negative_slope=0.2, inplace=False)
        
        # Store configuration
        self.num_features = self.embed_dim = embed_dim

        assert len(self_attn_block_style_list) == len(cross_attn_block_style_list) == depth
        
        # Build transformer decoder blocks
        self.blocks = TransformerDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            init_values=init_values,
            norm_layer=norm_layer, 
            act_layer=act_layer,
            self_attn_block_style_list=self_attn_block_style_list, 
            self_attn_combine_style=self_attn_combine_style,
            cross_attn_block_style_list=cross_attn_block_style_list, 
            cross_attn_combine_style=cross_attn_combine_style,
            k=k,
            proxy_obj_k=proxy_obj_k,
            mode=mode,
            bias_epsilon=bias_epsilon,
            scale_factor=scale_factor
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for different layer types."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, VecLinear):
            trunc_normal_(m.weight, std=.02)

    def forward(self, q, v, q_pos, v_pos, denoise_length=None,
                num_object_queries=None, num_proxy_queries=None):
        """
        Forward pass of PointTransformerDecoder.
        
        Args:
            q (torch.Tensor): Query tensor
            v (torch.Tensor): Value/key tensor
            q_pos (torch.Tensor): Query positions
            v_pos (torch.Tensor): Value positions
            denoise_length (int, optional): Length of denoising queries for masking
            
        Returns:
            torch.Tensor: Output query tensor after cross-attention processing
        """
        q = self.blocks(
            q, v, q_pos, v_pos, denoise_length=denoise_length,
            num_object_queries=num_object_queries,
            num_proxy_queries=num_proxy_queries,
        )
        return q


class PointTransformerEncoderEntry(PointTransformerEncoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))


class PointTransformerDecoderEntry(PointTransformerDecoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))


class VecDGCNN(nn.Module):
    def __init__(self, k=16, bias_epsilon=1e-6, rgb_dim=None):
        super().__init__()
        self.k = k
        self.rgb_dim = rgb_dim

        act_func = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        # Geometry branch
        self.conv1 = VecLinearNormalizeActivate(2, 32, mode="se3", act_func=act_func, bias_epsilon=bias_epsilon)
        self.conv2 = VecLinearNormalizeActivate(64, 64, mode="se3", act_func=act_func, bias_epsilon=bias_epsilon)
        self.conv3 = VecLinearNormalizeActivate(128, 64, mode="se3", act_func=act_func, bias_epsilon=bias_epsilon)
        self.conv4 = VecLinearNormalizeActivate(128, 128, mode="se3", act_func=act_func, bias_epsilon=bias_epsilon)

        self.pool1 = VecMaxPool(32, mode="se3", bias_epsilon=bias_epsilon)
        self.pool2 = VecMaxPool(64, mode="se3", bias_epsilon=bias_epsilon)
        self.pool3 = VecMaxPool(64, mode="se3", bias_epsilon=bias_epsilon)
        self.pool4 = VecMaxPool(128, mode="se3", bias_epsilon=bias_epsilon)

        self.num_features = 128

        # RGB branch, keeps [B, C_rgb, N]
        if rgb_dim is not None:
            self.rgb_mlp1 = self._make_rgb_mlp(rgb_dim)
            self.rgb_mlp2 = self._make_rgb_mlp(rgb_dim)
            self.rgb_mlp3 = self._make_rgb_mlp(rgb_dim)
            self.rgb_mlp4 = self._make_rgb_mlp(rgb_dim)
        else:
            self.rgb_mlp1 = None
            self.rgb_mlp2 = None
            self.rgb_mlp3 = None
            self.rgb_mlp4 = None

    @staticmethod
    def _make_rgb_mlp(dim):
        return nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.GELU(),
            nn.Conv1d(dim, dim, 1)
        )

    @staticmethod
    def _update_rgb(rgb_local, mlp):
        if rgb_local is None or mlp is None:
            return rgb_local
        return rgb_local + mlp(rgb_local)

    @staticmethod
    def fps_downsample(coor, x, num_group, rgb_local=None):
        """
        coor:      [B, 3, N]
        x:         [B, C, 3, N]
        rgb_local: [B, C_rgb, N] or None
        """

        xyz = coor.transpose(1, 2).contiguous()
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        B, C, D, N = x.shape
        x_reshaped = x.view(B, C * D, N)

        combined_x = torch.cat([coor, x_reshaped], dim=1)
        new_combined_x = pointnet2_utils.gather_operation(combined_x, fps_idx)

        new_coor = new_combined_x[:, :3, :]
        new_x = new_combined_x[:, 3:, :].view(B, C, D, num_group)

        new_rgb_local = None

        if rgb_local is not None:
            if rgb_local.dim() != 3:
                raise ValueError(f"rgb_local must be [B, C_rgb, N], got {rgb_local.shape}")

            if rgb_local.shape[0] != B or rgb_local.shape[2] != N:
                raise ValueError(
                    f"xyz and rgb_local are not aligned: "
                    f"xyz points={N}, rgb_local={rgb_local.shape}"
                )

            new_rgb_local = pointnet2_utils.gather_operation(
                rgb_local.contiguous(), fps_idx
            )

        return new_coor, new_x, new_rgb_local

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):
        """
        coor_q: [B, 3, M]
        x_q:    [B, C, 3, M]
        coor_k: [B, 3, N]
        x_k:    [B, C, 3, N]

        return: [B, 2C, 3, M, k]
        """

        bias = x_k.mean(dim=-1, keepdim=True)
        x_q = x_q - bias
        x_k = x_k - bias

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(-1)
        num_points_q = x_q.size(-1)

        with torch.no_grad():
            idx = knn_point(
                k,
                coor_k.transpose(-1, -2).contiguous(),
                coor_q.transpose(-1, -2).contiguous()
            )

            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k

            idx_base = torch.arange(
                batch_size, device=x_q.device
            ).view(-1, 1, 1) * num_points_k

            idx = (idx + idx_base).view(-1)

        num_dims = x_k.size(1)

        feature = x_k.permute(0, 3, 1, 2).contiguous()
        feature = feature.view(batch_size * num_points_k, num_dims, -1)[idx]
        feature = feature.view(batch_size, k, num_points_q, num_dims, -1)
        feature = feature.permute(0, 3, 4, 2, 1).contiguous()

        x_q = x_q.view(
            batch_size, num_dims, 3, num_points_q, 1
        ).expand(-1, -1, -1, -1, k)

        feature = torch.cat((feature - x_q, x_q), dim=1)
        feature = feature + torch.cat((bias, bias), dim=1).unsqueeze(-1)

        return feature

    def forward(self, x, num, rgb_local=None):
        """
        x:         [B, N, 3]
        rgb_local: [B, C_rgb, N] or None

        Example:
            x         = [40, 1024, 3]
            rgb_local = [40, 128, 1024]
        """

        has_rgb = rgb_local is not None

        if has_rgb:
            if self.rgb_dim is None:
                raise ValueError("rgb_local is provided, but rgb_dim=None.")

            if rgb_local.dim() != 3:
                raise ValueError(f"rgb_local must be [B, C_rgb, N], got {rgb_local.shape}")

            if rgb_local.shape[0] != x.shape[0] or rgb_local.shape[2] != x.shape[1]:
                raise ValueError(
                    f"rgb_local must correspond to xyz: "
                    f"xyz={x.shape}, rgb_local={rgb_local.shape}"
                )

            if rgb_local.shape[1] != self.rgb_dim:
                raise ValueError(
                    f"Expected rgb_dim={self.rgb_dim}, got {rgb_local.shape[1]}"
                )

        # xyz: [B,N,3] -> [B,3,N]
        coor = x.transpose(-1, -2).contiguous()

        # vector feature: [B,1,3,N]
        x = x.transpose(-1, -2).unsqueeze(1).contiguous()

        # Stage 1
        f = self.get_graph_feature(coor, x, coor, x)
        f = self.conv1(f)
        f = self.pool1(f)

        if has_rgb:
            rgb_local = self._update_rgb(rgb_local, self.rgb_mlp1)

        # FPS 1
        coor_q, f_q, rgb_local = self.fps_downsample(
            coor, f, num[0], rgb_local
        )

        # Stage 2
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.conv2(f)
        f = self.pool2(f)
        coor = coor_q

        if has_rgb:
            rgb_local = self._update_rgb(rgb_local, self.rgb_mlp2)

        # Stage 3
        f = self.get_graph_feature(coor, f, coor, f)
        f = self.conv3(f)
        f = self.pool3(f)

        if has_rgb:
            rgb_local = self._update_rgb(rgb_local, self.rgb_mlp3)

        # FPS 2
        coor_q, f_q, rgb_local = self.fps_downsample(
            coor, f, num[1], rgb_local
        )

        # Stage 4
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.conv4(f)
        f = self.pool4(f)

        if has_rgb:
            rgb_local = self._update_rgb(rgb_local, self.rgb_mlp4)

        # [B,3,M] -> [B,M,3]
        coor = coor_q.transpose(2, 1).contiguous()

        if has_rgb:
            return coor, f, rgb_local

        return coor, f

class SimpleRebuildFCLayer(nn.Module):
    """
    Simple Fully Connected Layer for Rebuilding Point Clouds

    Applies an MLP on concatenated global and local features.
    """
    
    def __init__(self, input_dims, step, hidden_dim=512, bias_epsilon=1e-6):
        super().__init__()
        self.input_dims = input_dims
        self.step = step
        self.pool = VecMaxPool(input_dims//2, mode="se3", bias_epsilon=bias_epsilon)
        self.layer = nn.Sequential(
            VecLinear(v_in=input_dims, v_out=hidden_dim, mode="so3", bias_epsilon=bias_epsilon),
            VecActivation(
                hidden_dim, 
                act_func=nn.LeakyReLU(negative_slope=0.2), 
                mode="so3", 
                bias_epsilon=bias_epsilon
            ),
            VecLinear(v_in=hidden_dim, v_out=step, mode="so3", bias_epsilon=bias_epsilon),
        )

    def forward(self, rec_feature):
        """
        Forward pass of SimpleRebuildFCLayer.
        
        Args:
            rec_feature (torch.Tensor): Input tensor with shape [B, C, 3, N]
            
        Returns:
            torch.Tensor: Rebuilt point cloud with shape [B, N, step, 3]
        """
        batch_size = rec_feature.size(0)

        # Extract global feature using pooling
        g_feature = self.pool(rec_feature.permute(0, 2, 3, 1))
        token_feature = rec_feature

        # Combine global and local features
        patch_feature = torch.cat(
            [
                g_feature.unsqueeze(1).expand(-1, token_feature.size(1), -1, -1),
                token_feature,
            ],
            dim=-2,
        )
        
        # Center features for translation invariance
        patch_feature = patch_feature - patch_feature.mean(dim=1, keepdim=True)
        
        # Generate rebuilt point cloud
        rebuild_pc = (
            self.layer(patch_feature.permute(0, 2, 3, 1))
            .permute(0, 3, 1, 2)
            .reshape(batch_size, -1, self.step, 3)
        )
        
        assert rebuild_pc.size(1) == rec_feature.size(1)
        
        return rebuild_pc


class QueryRanking(nn.Module):
    """
    Query Ranking Module for Point Cloud Processing.
    
    This module ranks queries based on their relevance using equivariant
    features and global context information.
    """
    
    def __init__(self, bias_epsilon=1e-6):
        """
        Initialize QueryRanking module.
        
        Args:
            mode (str): Equivariance mode, default "so3"
            bias_epsilon (float): Small constant for numerical stability
        """
        super().__init__()
        
        # Vector layers for SO(3)-equivariant processing
        self.fc_inv = VecLinear(1, 256, mode="so3", bias_epsilon=bias_epsilon)
        self.fc_O = VecLinear(1024, 256, mode="se3", bias_epsilon=bias_epsilon)
        self.fc_bias = VecLinear(1024, 256, mode="se3", bias_epsilon=bias_epsilon)
        
        # Scalar ranking network
        self.query_ranking = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, global_feature, x):
        """
        Forward pass of QueryRanking.
        
        Args:
            global_feature (torch.Tensor): Global feature tensor with shape [B, 1024, 3, 1]
            x (torch.Tensor): Input point coordinates with shape [B, N, 3]
            
        Returns:
            torch.Tensor: Ranking scores for each query point with shape [B, N, 1]
        """
        # Compute bias term from global features
        bias = self.fc_bias(global_feature)  # [B, 256, 3, 1]
        
        # Transform global features to SO(3) space
        z_so3 = self.fc_O(global_feature).unsqueeze(-1) - bias.unsqueeze(-1)  # [B, 256, 3, 1, 1]
        
        # Center input points for translation invariance
        x = x - x.mean(dim=1, keepdim=True)  # [B, N, 3]
        
        # Process each point with vector layers
        v_inv_per_point = self.fc_inv(
            x.unsqueeze(-2).permute(0, 2, 3, 1)
        )  # [B, 256, 3, N]
        
        # Compute invariant features through dot product
        v_inv_per_point = (v_inv_per_point * z_so3).sum(-2).permute(0, 2, 1)  # [B, N, 256]
        
        # Normalize features for stability
        v_inv_per_point = v_inv_per_point / (
            v_inv_per_point.norm(dim=-1, keepdim=True) + 1e-6
        )
        
        # Generate ranking scores
        ranking_scores = self.query_ranking(v_inv_per_point)  # [B, N, 1]
        
        return ranking_scores

######################################## PCTransformer ########################################   
class PCTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_config = config.encoder_config
        decoder_config = config.decoder_config
        self.center_num = getattr(config, 'center_num', [512, 128])
        self.denoise_length = getattr(config, 'denoise_length', 64)
        self.query_selection = getattr(config, 'query_selection', True)
        self.mode = getattr(config, 'mode', "sim3")
        self.encoder_type = config.encoder_type
        self.bias_epsilon = getattr(config, 'bias_epsilon', 1e-6)
        self.rgb_dim = getattr(config, 'rgb_dim', None)

        self.skeleton_num = getattr(config, 'skeleton_num', 36)

        
        assert self.encoder_type == 'vecgraph', f'unexpected encoder_type {self.encoder_type}'
        act_func = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.num_query = query_num = config.num_query
        global_feature_dim = config.global_feature_dim

        if self.rgb_dim is not None:
            self.rgb_global_proj = nn.Sequential(
                nn.Conv1d(self.rgb_dim, global_feature_dim, 1),
                nn.GELU(),
                nn.Conv1d(global_feature_dim, global_feature_dim, 1)
            )
            self.global_lin_ori = VecLinear(global_feature_dim, global_feature_dim, mode=self.mode, bias_epsilon=self.bias_epsilon)
        else:
            self.rgb_global_proj = None
        # Base encoder
        if self.encoder_type == 'vecgraph':
            self.grouper = VecDGCNN(bias_epsilon=self.bias_epsilon, rgb_dim=self.rgb_dim)
        else:
            raise NotImplementedError(f'encoder_type {self.encoder_type} not implemented')
            
        self.pos_embed = nn.Sequential(
            VecLinear(v_in=1, v_out=128, mode="so3", bias_epsilon=self.bias_epsilon),
            VecActivation(128, act_func=act_func, mode="so3", bias_epsilon=self.bias_epsilon),
            VecLinear(v_in=128, v_out=encoder_config.embed_dim, mode="so3", bias_epsilon=self.bias_epsilon),
        )
        
        self.input_proj = nn.Sequential(
            VecLinear(v_in=self.grouper.num_features, v_out=512, mode=self.mode, bias_epsilon=self.bias_epsilon),
            VecActivation(512, act_func=act_func, mode=self.mode, bias_epsilon=self.bias_epsilon),
            VecLinear(v_in=512, v_out=encoder_config.embed_dim, mode=self.mode, bias_epsilon=self.bias_epsilon),
        )

        self.skeleton_query = nn.Parameter(torch.empty(1, global_feature_dim))
        nn.init.xavier_normal_(self.skeleton_query)

        self.encoder = PointTransformerEncoderEntry(encoder_config)
        
        self.increase_dim = nn.Sequential(
            VecLinear(v_in=encoder_config.embed_dim, v_out=1024, mode=self.mode, bias_epsilon=self.bias_epsilon),
            VecActivation(1024, act_func=act_func, mode=self.mode, bias_epsilon=self.bias_epsilon),
            VecLinear(v_in=1024, v_out=global_feature_dim, mode=self.mode, bias_epsilon=self.bias_epsilon),
        )
        self.pool = VecMaxPool(global_feature_dim, mode=self.mode, bias_epsilon=self.bias_epsilon)
        self.coarse_pred = nn.Sequential(
            VecLinear(v_in=global_feature_dim, v_out=1024, mode=self.mode, bias_epsilon=self.bias_epsilon),
            VecActivation(1024, act_func=act_func, mode=self.mode, bias_epsilon=self.bias_epsilon),
            VecLinear(v_in=1024, v_out=query_num, mode=self.mode, bias_epsilon=self.bias_epsilon),
        )

        self.global_lin_ori2 = VecLinear(global_feature_dim, global_feature_dim, mode=self.mode, bias_epsilon=self.bias_epsilon)
        self.skeleton_pred = nn.Sequential(
            VecLinear(v_in=global_feature_dim, v_out=1024, mode=self.mode, bias_epsilon=self.bias_epsilon),
            VecActivation(1024, act_func=act_func, mode=self.mode, bias_epsilon=self.bias_epsilon),
            VecLinear(v_in=1024, v_out=self.skeleton_num, mode=self.mode, bias_epsilon=self.bias_epsilon),
        )
        
        self.mlp_query = nn.Sequential(
            VecLinear(v_in=global_feature_dim + 1, v_out=1024, mode=self.mode, bias_epsilon=self.bias_epsilon),
            VecActivation(1024, act_func=act_func, mode=self.mode, bias_epsilon=self.bias_epsilon),
            VecLinear(v_in=1024, v_out=1024, mode=self.mode, bias_epsilon=self.bias_epsilon),
            VecActivation(1024, act_func=act_func, mode=self.mode, bias_epsilon=self.bias_epsilon),
            VecLinear(
                v_in=1024, v_out=decoder_config.embed_dim, mode=self.mode, bias_epsilon=self.bias_epsilon
            )
        )

        if decoder_config.embed_dim == encoder_config.embed_dim:
            self.mem_link = nn.Identity()
        else:
            self.mem_link = VecLinear(
                v_in=encoder_config.embed_dim,
                v_out=decoder_config.embed_dim,
                mode=self.mode,
                bias_epsilon=self.bias_epsilon,
            )

        self.decoder = PointTransformerDecoderEntry(decoder_config)
        self.query_ranking = QueryRanking(self.bias_epsilon)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, xyz, rgb_local=None):
        # Extract features using graph-based encoder
        if rgb_local is not None:
            coor, f, rgb_local = self.grouper(xyz, self.center_num, rgb_local)

            # rgb_local: [B,M,128] -> [B,128,1,M]
            rgb_weight = rgb_local.unsqueeze(2).contiguous()

            # Preserve translation-related component
            bias = f.mean(dim=-1, keepdim=True)
            f = (f - bias) * rgb_weight + bias
        else:
            coor, f = self.grouper(xyz, self.center_num)
        # Project input features to embedding dimension
        x = self.input_proj(f)
        
        # Add positional embedding (centered for translation invariance)
        coor_mean = coor.mean(dim=1, keepdim=True)
        pe = self.pos_embed(
            coor.permute(0, 2, 1).unsqueeze(1).contiguous() - 
            coor_mean.permute(0, 2, 1).unsqueeze(1).contiguous()
        )
        x = (x + pe).permute(0, 3, 1, 2).contiguous()
        
        # Apply transformer encoder
        x = self.encoder(x, coor)
        
        # Extract global features
        global_feature = self.increase_dim(x.permute(0, 2, 3, 1).contiguous())
        global_feature = self.pool(global_feature)

        # Global RGB-geometry fusion
        if rgb_local is not None:
            rgb_global = rgb_local.mean(dim=-1, keepdim=True)
            # [B,128,1]

            rgb_global = self.rgb_global_proj(rgb_global)
            # [B,512,1]

            rgb_weight = rgb_global
            # [B,512,1,1]
            global_origin = self.global_lin_ori(global_feature)

            global_feature = (global_feature - global_origin) * rgb_weight + global_origin

        # Predict coarse point cloud
        coarse = self.coarse_pred(global_feature)

        global_origin2 = self.global_lin_ori2(global_feature)
        global_feature2 = (global_feature - global_origin2) * self.skeleton_query.unsqueeze(-1) + global_origin2

        skeleton_points = self.skeleton_pred(global_feature2)
        
        # Apply query selection if enabled
        if self.query_selection:
            coarse_inp = misc.fps(xyz.contiguous(), self.num_query // 2)
            coarse = torch.cat([coarse, coarse_inp], dim=1)
            query_ranking = self.query_ranking(global_feature, coarse)
            idx = torch.argsort(query_ranking, dim=1, descending=True)
            coarse = torch.gather(
                coarse, 1, idx[:, :self.num_query].expand(-1, -1, coarse.size(-1))
            )

        # Prepare memory features for decoder
        mem = self.mem_link(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

        # Query order: [coarse, skeleton, denoise]. Denoise queries must stay last.
        query_points = torch.cat([coarse, skeleton_points], dim=1)
        global_query = torch.cat([
            global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1, -1),
            global_feature2.unsqueeze(1).expand(-1, skeleton_points.size(1), -1, -1),
        ], dim=1)
        denoise_length = self.denoise_length if self.training and self.denoise_length > 0 else None
        if denoise_length:
            noisy_points = misc.jitter_points(misc.fps(xyz.contiguous(), denoise_length))
            query_points = torch.cat([query_points, noisy_points], dim=1)
            global_query = torch.cat([
                global_query, global_feature.unsqueeze(1).expand(-1, denoise_length, -1, -1)
            ], dim=1)

        q = self.mlp_query(torch.cat([global_query, query_points.unsqueeze(2)], dim=-2)
                           .permute(0, 2, 3, 1).contiguous())
        q = self.decoder(q=q.permute(0, 3, 1, 2).contiguous(), v=mem,
                         q_pos=query_points, v_pos=coor, denoise_length=denoise_length,
                         num_object_queries=self.num_query, num_proxy_queries=self.skeleton_num)
        return q, query_points, mem, coor, denoise_length or 0


class sim3Reconstructor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pts_per_kpt = cfg.pts_per_kpt
        self.ndim = cfg.ndim
        self.mode = getattr(cfg, 'mode', 'sim3')
        self.bias_epsilon = getattr(cfg, 'bias_epsilon', 1e-6)

        self.increase_dim = nn.Sequential(
            VecLinear(self.ndim, 1024, mode=self.mode, bias_epsilon=self.bias_epsilon),
            VecActivation(1024, act_func=nn.LeakyReLU(0.2), mode=self.mode, bias_epsilon=self.bias_epsilon),
            VecLinear(1024, 1024, mode=self.mode, bias_epsilon=self.bias_epsilon),
        )

        self.pool = VecMaxPool(1024, mode=self.mode, bias_epsilon=self.bias_epsilon)

        self.reduce_map = nn.Sequential(
            VecLinear(self.ndim + 1025, self.ndim, mode=self.mode, bias_epsilon=self.bias_epsilon),
        )

        self.decode_head = SimpleRebuildFCLayer(
            2 * self.ndim, step=self.pts_per_kpt, bias_epsilon=self.bias_epsilon
        )

    def forward(self, q, kpt_3d):
        """
        Args:
            q: (B, M, C, 3)
            kpt_3d: (B, M, 3)

        Returns:
            recon_model: (B, 3, M * pts_per_kpt)
            recon_delta: (B, M * pts_per_kpt, 3)
        """
        B, M = q.shape[:2]

        # Extract global features: (B, 1024, 3)
        global_feature = self.increase_dim(q.permute(0, 2, 3, 1).contiguous())
        global_feature = self.pool(global_feature)

        # Combine global features, local features and coarse points
        rebuild_feature = torch.cat([
            global_feature.unsqueeze(1).expand(-1, M, -1, -1),
            q,
            kpt_3d.unsqueeze(2),
        ], dim=2)  # (B, M, 1024 + C + 1, 3)

        # Reduce to C point-type channels; decode_head converts them to vector offsets.
        rebuild_feature = self.reduce_map(
            rebuild_feature.permute(0, 2, 3, 1).contiguous()
        ).permute(0, 3, 1, 2).contiguous()

        # Predict relative coordinates: (B, M, pts_per_kpt, 3)
        relative_xyz = self.decode_head(rebuild_feature)

        # Reconstruct point cloud
        rebuild_points = relative_xyz + kpt_3d.unsqueeze(2)

        recon_model = rebuild_points.reshape(B, -1, 3).transpose(1, 2).contiguous()
        recon_delta = relative_xyz.reshape(B, -1, 3).contiguous()

        return recon_model, recon_delta

@MODELS.register_module()
class SIMECO(nn.Module):
    """
    SIMECO Model

    A point cloud completion model that integrates a transformer-based encoder-decoder
    architecture with a reconstruction head for generating dense point clouds.
    """
    def __init__(self, config, **kwargs):
        super().__init__()
        
        # Configuration parameters
        self.trans_dim = config.decoder_config.embed_dim
        self.num_query = config.num_query
        self.num_points = getattr(config, 'num_points', None)
        self.bias_epsilon = getattr(config, 'bias_epsilon', 1e-6)
        self.decoder_type = config.decoder_type
        self.encoder_type = config.encoder_type
        self.mode = getattr(config, 'mode', "sim3")
        self.denoise_length = getattr(config, 'denoise_length', 64)
        
        # Validate decoder type
        assert self.decoder_type in ['fold', 'fc'], f'unexpected decoder_type {self.decoder_type}'
        
        # Initialize base model
        self.fold_step = 8
        self.base_model = PCTransformer(config)
        
        # Configure decoder head based on num_points
        if self.num_points is not None:
            self.factor = self.num_points // self.num_query
            assert self.num_points % self.num_query == 0
            self.decode_head = SimpleRebuildFCLayer(
                self.trans_dim * 2, 
                step=self.num_points // self.num_query, 
                bias_epsilon=self.bias_epsilon
            )
        else:
            self.factor = self.fold_step**2
            self.decode_head = SimpleRebuildFCLayer(
                self.trans_dim * 2, 
                step=self.fold_step**2, 
                bias_epsilon=self.bias_epsilon
            )
        
        # Feature enhancement layers
        self.increase_dim = nn.Sequential(
            VecLinear(self.trans_dim, 1024, mode=self.mode, bias_epsilon=self.bias_epsilon),
            VecActivation(
                1024, 
                act_func=nn.LeakyReLU(negative_slope=0.2), 
                mode=self.mode, 
                bias_epsilon=self.bias_epsilon
            ),
            VecLinear(1024, 1024, mode=self.mode, bias_epsilon=self.bias_epsilon),
        )
        
        # Global pooling layer
        self.pool = VecMaxPool(1024, mode=self.mode, bias_epsilon=self.bias_epsilon)
        
        # Feature dimension reduction layer
        self.reduce_map = nn.Sequential(
            VecLinear(
                self.trans_dim + 1025, 
                self.trans_dim, 
                mode=self.mode, 
                bias_epsilon=self.bias_epsilon
            )
        )
        
        # Initialize loss function
        self.build_loss_func()

    def build_loss_func(self):
        """Initialize the loss function for training."""
        # Chamfer is only required by the full SIMECO completion wrapper.  Keep
        # the binary extension out of the PCTransformer import path because it
        # is tied to the PyTorch ABI used when it was compiled.
        from extensions.chamfer_dist import ChamferDistanceL1
        self.loss_func = ChamferDistanceL1()

    def get_loss(self, ret, gt):
        """
        Compute training losses.
        
        Args:
            ret: Model predictions (pred_coarse, denoised_coarse, denoised_fine, pred_fine)
            gt: Ground truth point cloud
            
        Returns:
            loss_denoised: Denoising task loss
            loss_recon: Reconstruction task loss
        """
        pred_coarse, denoised_coarse, denoised_fine, pred_fine = ret
        assert pred_fine.size(1) == gt.size(1)

        # Compute denoising loss
        idx = knn_point(self.factor, gt, denoised_coarse)  # B n k 
        denoised_target = index_points(gt, idx)  # B n k 3 
        denoised_target = denoised_target.reshape(gt.size(0), -1, 3)
        assert denoised_target.size(1) == denoised_fine.size(1)
        loss_denoised = self.loss_func(denoised_fine, denoised_target)
        loss_denoised = loss_denoised * 0.5

        # Compute reconstruction losses
        loss_coarse = self.loss_func(pred_coarse, gt)
        loss_fine = self.loss_func(pred_fine, gt)
        loss_recon = loss_coarse + loss_fine

        return loss_denoised, loss_recon

    def forward(self, xyz):
        """
        Forward pass of SIMECO model.
        
        Args:
            xyz: Input point cloud [B, N, 3]
            
        Returns:
            Training: (pred_coarse, denoised_coarse, denoised_fine, pred_fine)
            Inference: (coarse_point_cloud, rebuild_points)
        """
        # Get features from base transformer model
        q, coarse_point_cloud, _, _, denoise_length = self.base_model(xyz)  # B M C 3 and B M 3
        B, M, C, _ = q.shape
        
        # Extract global features
        global_feature = self.increase_dim(q.permute(0, 2, 3, 1).contiguous())
        global_feature = self.pool(global_feature)
        
        # Combine features for rebuilding
        rebuild_feature = torch.cat(
            [
                global_feature.unsqueeze(-3).expand(-1, M, -1, -1),
                q,
                coarse_point_cloud.unsqueeze(-2),
            ],
            dim=-2,
        )  # B M (1024 + C + 1) 3

        # Apply feature reduction
        rebuild_feature = self.reduce_map(
            rebuild_feature.permute(0, 2, 3, 1).contiguous()
        ).permute(0, 3, 1, 2).contiguous()
        
        # Generate relative coordinates and add to coarse points
        relative_xyz = self.decode_head(rebuild_feature)
        rebuild_points = relative_xyz + coarse_point_cloud.unsqueeze(-2)  # B M S 3

        if self.training:
            # Split predictions for training tasks
            pred_fine = rebuild_points[:, :-denoise_length].reshape(B, -1, 3).contiguous()
            pred_coarse = coarse_point_cloud[:, :-denoise_length].contiguous()

            denoised_fine = rebuild_points[:, -denoise_length:].reshape(B, -1, 3).contiguous()
            denoised_coarse = coarse_point_cloud[:, -denoise_length:].contiguous()
            
            # Validate output dimensions
            assert pred_fine.size(1) == self.num_query * self.factor
            assert pred_coarse.size(1) == self.num_query

            ret = (pred_coarse, denoised_coarse, denoised_fine, pred_fine)
            return ret

        else:
            # Inference mode - no denoising
            assert denoise_length == 0
            rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()  # B N 3

            # Validate output dimensions
            assert rebuild_points.size(1) == self.num_query * self.factor
            assert coarse_point_cloud.size(1) == self.num_query
            
            ret = (coarse_point_cloud, rebuild_points)
            return ret
