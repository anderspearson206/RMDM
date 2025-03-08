"""
Radio Map Generation Neural Network Models
=======================================

This module implements various neural network architectures for radio map generation
and wireless signal strength prediction. The main components include:

1. RadioMapGenerationModel: A UNet-based architecture for generating radio maps
2. Attention mechanisms for capturing spatial signal correlations
3. Multi-scale feature processing for different propagation characteristics
4. Residual connections for preserving signal details

The models are designed to predict radio signal strength distributions in wireless
environments, taking into account various propagation phenomena like path loss,
shadowing, and multi-path effects.
"""

from abc import abstractmethod
import math
import numpy as np
import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from fp16_util import convert_module_to_f16, convert_module_to_f32
from copy import deepcopy
from utils import softmax_helper,sigmoid_helper
from utils import InitWeights_He
from batchgenerators.augmentations.utils import pad_nd_image
from utils import no_op
from utils import to_cuda, maybe_to_torch
from scipy.ndimage.filters import gaussian_filter
from typing import Union, Tuple, List
from torch.cuda.amp import autocast
from nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    layer_norm,
)


class AttentionPool2d(nn.Module):
    """
    Attention pooling layer for processing spatial radio signal features.
    Adapted from CLIP but modified for radio map generation tasks.
    
    This layer helps capture long-range dependencies in signal propagation patterns.
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Base class for modules that process radio signal features with temporal dynamics.
    
    In radio map generation, temporal aspects can represent:
    - Signal variations over time
    - Dynamic channel conditions
    - Mobile transmitter/receiver scenarios
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Process input features with temporal embeddings.
        
        Args:
            x: Input radio signal features
            emb: Temporal embedding representing dynamic conditions
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential container adapted for radio map generation tasks.
    Handles both spatial features and temporal dynamics of wireless channels.
    
    This module is particularly useful for modeling time-varying channel conditions
    and dynamic radio environments.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    Upsampling layer for radio map feature processing.
    
    Used to generate higher resolution radio maps from lower resolution features.
    Important for:
    - Detailed signal strength predictions
    - Fine-grained coverage analysis
    - High-resolution path loss mapping
    
    Args:
        channels: Number of feature channels
        use_conv: Whether to apply convolution after upsampling
        dims: Dimensionality of the radio map (typically 2 for 2D maps)
        out_channels: Optional different number of output channels
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Downsampling layer for radio map feature processing.
    
    Used to capture larger-scale propagation effects by reducing spatial resolution.
    Important for:
    - Large-scale path loss modeling
    - Coverage area analysis
    - Multi-scale signal propagation effects
    
    Args:
        channels: Number of feature channels
        use_conv: Whether to use convolution for downsampling
        dims: Dimensionality of the radio map
        out_channels: Optional different number of output channels
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

def conv_bn(inp, oup, stride):
    """
    Convolution-BatchNorm block for radio map feature extraction.
    Provides basic feature processing with normalization.
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
        )

def conv_dw(inp, oup, stride):
    """
    Depthwise separable convolution block for efficient radio map processing.
    Reduces computational cost while maintaining feature quality.
    """
    return nn.Sequential(
        # Depthwise conv
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        # Pointwise conv
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

class MobBlock(nn.Module):
    """
    Mobile-optimized block for radio map generation.
    Provides efficient feature extraction at different scales.
    
    This block is particularly useful for:
    - Resource-efficient radio map generation
    - Mobile device deployment
    - Real-time signal strength prediction
    """
    def __init__(self,ind):
        super().__init__()


        if ind == 0:
            self.stage = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 1),
            conv_dw(128, 128, 1)
        )
        elif ind == 1:
            self.stage  = nn.Sequential(
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1)
        )
        elif ind == 2:
            self.stage = nn.Sequential(
            conv_dw(256, 256, 2),
            conv_dw(256, 256, 1)
            )
        else:
            self.stage = nn.Sequential(
                conv_dw(256, 512, 2),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1)
            )

    def forward(self,x):
        return self.stage(x)



class ResBlock(TimestepBlock):
    """
    Residual block specialized for radio map generation.
    
    This block is crucial for:
    - Preserving fine-grained signal propagation details
    - Modeling complex multi-path effects
    - Maintaining signal strength gradients
    
    The residual connection helps prevent loss of important propagation information
    through deep networks, which is essential for accurate radio map prediction.
    
    Args:
        channels: Number of input feature channels
        emb_channels: Temporal embedding channels for dynamic conditions
        dropout: Dropout rate for regularization
        out_channels: Optional output channel count
        use_conv: Use spatial conv for channel change
        dims: Dimensionality of radio map
        use_checkpoint: Use gradient checkpointing
        up: Use for upsampling features
        down: Use for downsampling features
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    Legacy version of QKV attention for radio map feature processing.
    
    This version splits heads before splitting QKV, which can be more
    memory efficient for processing large radio maps.
    """

    def __init__(self, n_heads):
        """
        Initialize QKV attention module.
        
        Args:
            n_heads: Number of attention heads for multi-head attention
        """
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply legacy QKV attention mechanism.
        
        Args:
            qkv: Input features [N x (H * 3 * C) x T]
            
        Returns:
            torch.Tensor: Processed features with attention
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        """
        Count floating point operations for attention.
        
        This is useful for:
        - Performance optimization
        - Hardware resource planning
        - Model complexity analysis
        """
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    Query-Key-Value attention module for radio map feature processing.
    
    This attention mechanism helps capture:
    - Long-range signal propagation patterns
    - Spatial correlations in signal strength
    - Complex interference patterns
    
    The attention is computed in a way that preserves the spatial
    relationships important for radio signal propagation modeling.
    """

    def __init__(self, n_heads):
        """
        Initialize QKV attention module.
        
        Args:
            n_heads: Number of attention heads for multi-head attention
        """
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention to input features.
        
        Args:
            qkv: Combined query, key, value tensor [N x (3 * H * C) x T]
            
        Returns:
            torch.Tensor: Attended features capturing spatial relationships
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


class FFParser(nn.Module):
    """
    Fourier Feature Parser for radio signal processing.
    
    Processes radio signal features in the frequency domain, which is particularly 
    useful for:
    - Analyzing signal frequency components
    - Modeling periodic interference patterns
    - Capturing large-scale spatial correlations
    
    Args:
        dim: Feature dimension
        h: Height of the feature map
        w: Width of the feature map
    """
    def __init__(self, dim, h=128, w=65):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, C, H, W = x.shape
        assert H == W, "height and width are not equal"
        if spatial_size is None:
            a = b = H
        else:
            a, b = spatial_size

        # x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')

        x = x.reshape(B, C, H, W)

        return x


class RadioMapUNet(SegmentationNetwork):
    """
    Advanced UNet architecture specialized for radio map generation and signal strength prediction.
    
    This network is designed for various wireless communication scenarios:
    - Large-scale path loss prediction
    - Indoor/outdoor signal coverage mapping
    - Interference pattern analysis
    - Multi-transmitter scenarios
    
    The architecture is optimized for:
    - Multi-scale signal propagation effects
    - Different environmental conditions
    - Various wireless network deployments
    
    Key features:
    - Flexible depth and width configuration
    - Support for 2D and 3D radio maps
    - Efficient memory usage
    - Optional deep supervision for better gradient flow
    """

    # Configuration for 3D radio environments (e.g., multi-floor buildings)
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    # Configuration for 2D radio maps (e.g., single floor or outdoor)
    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    # Memory estimation parameters
    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        
        # ... rest of the existing initialization code ...

    def forward(self, x, hs = None):
        """
        Forward pass of the radio map generation network.
        
        Args:
            x: Input tensor containing signal measurements or features
            hs: Optional hierarchical features from external sources
            
        Returns:
            tuple: (embedding, predictions)
                - embedding: Learned signal propagation features
                - predictions: Predicted radio map values
        """
        skips = []
        seg_outputs = []
        anch_outputs = []
        
        # Encoder path - extract multi-scale propagation features
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)
            if hs:
                # Process external hierarchical features
                h = hs.pop(0)
                ddims = h.size(1)
                h = self.conv_trans_blocks_a[d](h)
                h = self.ffparser[d](h)  # Process in frequency domain
                ha = self.conv_trans_blocks_b[d](h)
                hb = th.mean(h,(2,3))
                hb = hb[:,:,None,None]
                x = x * ha * hb  # Modulate features with external information

        # Bottleneck processing
        x = self.conv_blocks_context[-1](x)
        emb = conv_nd(2, x.size(1), 512, 1).to(device = x.device)(x)

        # Decoder path - reconstruct radio map
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = th.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            if self._deep_supervision:
                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
            if self.anchor_out and (not self._deep_supervision):
                anch_outputs.append(x)

        if not seg_outputs:
            seg_outputs.append(self.final_nonlin(self.seg_outputs[0](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        if self.anchor_out:
            return tuple([i(j) for i, j in
                                        zip(list(self.upscale_logits_ops)[::-1], anch_outputs[:-1][::-1])]),seg_outputs[-1]
        else:
            return emb, seg_outputs[-1]

    def forward_sample(self, x, input, t):
        """
        Forward pass with additional input features and temporal condition.
        
        This method is specifically designed for radio map generation tasks where
        we need to consider:
        - Current signal measurements (x)
        - Additional environmental features (input)
        - Temporal dynamics (t)
        
        Args:
            x: Current signal measurements [B, C, H, W]
            input: Additional input features [B, C', H, W]
            t: Temporal condition [B]
            
        Returns:
            torch.Tensor: Generated radio map
        """
        # Combine signal measurements with additional features
        combined_input = torch.cat([x, input], dim=1)
        
        # Get temporal embeddings
        emb = self.time_embed(timestep_embedding(t, self.model_channels))
        
        # Process through the network
        return self.forward(combined_input, emb)[1]  # Return predictions only

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        Estimate approximate VRAM consumption for radio map generation.
        
        This is useful for:
        - Planning hardware requirements
        - Optimizing batch sizes
        - Managing multi-GPU training
        
        Args:
            patch_size: Size of radio map patches
            num_pool_per_axis: Pooling operations per axis
            base_num_features: Initial feature maps
            max_num_features: Maximum feature maps
            num_modalities: Number of input modalities (e.g., RSSI, SNR)
            num_classes: Number of output classes
            pool_op_kernel_sizes: Pooling kernel sizes
            deep_supervision: Whether using deep supervision
            conv_per_stage: Convolutions per stage
            
        Returns:
            int: Approximate VRAM requirement in bytes
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp


class ConvDropoutNormNonlin(nn.Module):
    """
    Basic convolutional block for radio map feature extraction.
    
    This block combines:
    - Convolution for feature extraction
    - Dropout for regularization
    - Normalization for stable training
    - Non-linear activation
    
    Particularly useful for:
    - Extracting local signal propagation patterns
    - Maintaining stable gradients
    - Preventing overfitting to noise in signal measurements
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        """
        Initialize the convolutional block.
        
        Args:
            input_channels: Number of input feature channels
            output_channels: Number of output feature channels
            conv_op: Convolution operation to use
            conv_kwargs: Convolution parameters
            norm_op: Normalization operation
            norm_op_kwargs: Normalization parameters
            dropout_op: Dropout operation
            dropout_op_kwargs: Dropout parameters
            nonlin: Non-linear activation
            nonlin_kwargs: Activation parameters
        """
        super().__init__()
        
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        """
        Forward pass through the convolutional block.
        
        Args:
            x: Input features
            
        Returns:
            torch.Tensor: Processed features
        """
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    """
    Variant of the basic convolutional block with different operation order.
    
    This version applies normalization after activation, which can be
    beneficial for certain radio map generation scenarios.
    """
    
    def forward(self, x):
        """
        Forward pass with modified operation order.
        
        Args:
            x: Input features
            
        Returns:
            torch.Tensor: Processed features
        """
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))


class StackedConvLayers(nn.Module):
    """
    Multiple stacked convolutional layers for deep feature extraction.
    
    This module is useful for:
    - Building deep feature hierarchies
    - Capturing multi-scale signal propagation effects
    - Learning complex spatial patterns in radio maps
    """

    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, 
                 basic_block=ConvDropoutNormNonlin):
        """
        Initialize stacked convolutional layers.
        
        Args:
            input_feature_channels: Initial number of channels
            output_feature_channels: Final number of channels
            num_convs: Number of convolutional layers to stack
            conv_op: Convolution operation
            conv_kwargs: Convolution parameters
            norm_op: Normalization operation
            norm_op_kwargs: Normalization parameters
            dropout_op: Dropout operation
            dropout_op_kwargs: Dropout parameters
            nonlin: Non-linear activation
            nonlin_kwargs: Activation parameters
            first_stride: Optional stride for first layer
            basic_block: Basic building block to use
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_convs):
            if i == 0:
                if first_stride is not None:
                    self.layers.append(basic_block(input_feature_channels, output_feature_channels,
                                                 conv_op=conv_op, conv_kwargs=conv_kwargs,
                                                 norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                                 dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                                                 nonlin=nonlin, nonlin_kwargs=nonlin_kwargs))
                else:
                    self.layers.append(basic_block(input_feature_channels, output_feature_channels,
                                                 conv_op=conv_op, conv_kwargs=conv_kwargs,
                                                 norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                                 dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                                                 nonlin=nonlin, nonlin_kwargs=nonlin_kwargs))
            else:
                self.layers.append(basic_block(output_feature_channels, output_feature_channels,
                                             conv_op=conv_op, conv_kwargs=conv_kwargs,
                                             norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                             dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                                             nonlin=nonlin, nonlin_kwargs=nonlin_kwargs))

    def forward(self, x):
        """
        Forward pass through the stacked convolutional layers.
        
        Args:
            x: Input features
            
        Returns:
            torch.Tensor: Processed features
        """
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    """
    Example usage of RadioMapUNet for radio map generation
    """
    image_size = 256
    in_ch = 3
    num_channels = 128  # Base number of channels
    channel_mult = (1, 1, 2, 2, 4, 4)  # Channel multipliers for multi-scale features
    num_res_blocks = 2  # Number of residual blocks per scale
    class_cond = False
    NUM_CLASSES = 2  # Number of signal strength levels to predict
    attention_ds = [32]  # Resolution at which to apply attention
    dropout = 0.0
    use_checkpoint = False  # Memory optimization
    use_fp16 = False  # Precision setting
    num_heads = 1  # Number of attention heads
    num_head_channels = -1  # Channels per attention head
    num_heads_upsample = -1  # Attention heads in upsampling
    use_scale_shift_norm = False  # Normalization setting
    resblock_updown = False  # Residual connections in scaling
    use_new_attention_order = False  # Attention pattern setting

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    # Initialize radio map generation model
    model = RadioMapUNet(
            image_size=image_size,
            in_channels=in_ch,
            model_channels=num_channels,
            out_channels=1,  # Single channel output for signal strength
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(NUM_CLASSES if class_cond else None),
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
        )
    model.to(device)

    # Prepare example input data
    x = th.randn(1, 1, 256, 256).to(device)  # Signal measurement input
    input = th.randn(1, 2, 256, 256).to(device)  # Additional features
    t = torch.rand(x.size(0)).to(device)  # Temporal condition
    
    # Generate radio map
    with th.no_grad():
        out = model.forward_sample(x, input, t)
        print("Output radio map shape:", out.shape)


