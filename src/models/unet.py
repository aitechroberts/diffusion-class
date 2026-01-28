"""
U-Net Architecture for Diffusion Models

In this file, you should implements a U-Net architecture suitable for DDPM.

Architecture Overview:
    Input: (batch_size, channels, H, W), timestep
    
    Encoder (Downsampling path)

    Middle
    
    Decoder (Upsampling path)
    
    Output: (batch_size, channels, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .blocks import (
    TimestepEmbedding,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    GroupNorm32,
)


class UNet(nn.Module):
    """
    TODO: design your own U-Net architecture for diffusion models.

    Args:
        in_channels: Number of input image channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        base_channels: Base channel count (multiplied by channel_mult at each level)
        channel_mult: Tuple of channel multipliers for each resolution level
                     e.g., (1, 2, 4, 8) means channels are [C, 2C, 4C, 8C]
        num_res_blocks: Number of residual blocks per resolution level
        attention_resolutions: Resolutions at which to apply self-attention
                              e.g., [16, 8] applies attention at 16x16 and 8x8
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_scale_shift_norm: Whether to use FiLM conditioning in ResBlocks
    
    Example:
        >>> model = UNet(
        ...     in_channels=3,
        ...     out_channels=3, 
        ...     base_channels=128,
        ...     channel_mult=(1, 2, 2, 4),
        ...     num_res_blocks=2,
        ...     attention_resolutions=[16, 8],
        ... )
        >>> x = torch.randn(4, 3, 64, 64)
        >>> t = torch.randint(0, 1000, (4,))
        >>> out = model(x, t)
        >>> out.shape
        torch.Size([4, 3, 64, 64])
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        num_heads: int = 4,
        dropout: float = 0.1,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm
        # Pro tips: remember to take care of the time embeddings!
        # Time embedding
        time_embed_dim = base_channels * 4
        self.time_embed = TimestepEmbedding(time_embed_dim)
        
        # Input convolution
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Calculate channel counts at each level
        channels = [base_channels * mult for mult in channel_mult]
        
        # Encoder (downsampling path)
        # encoder_blocks is a list of levels, each level contains multiple ResBlock(+Attention) groups
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        
        current_channels = base_channels
        for level, mult in enumerate(channel_mult):
            out_channels_level = base_channels * mult
            
            # Resolution at this level (assuming 64x64 input)
            resolution = 64 // (2 ** level)
            
            # Collect all blocks for this level
            level_blocks = []
            for _ in range(num_res_blocks):
                layers = [ResBlock(
                    current_channels,
                    out_channels_level,
                    time_embed_dim,
                    dropout=dropout,
                    use_scale_shift_norm=use_scale_shift_norm,
                )]
                
                # Add attention if this resolution is in attention_resolutions
                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(out_channels_level, num_heads=num_heads))
                
                level_blocks.append(nn.ModuleList(layers))
                current_channels = out_channels_level
            
            # Add all blocks for this level as a group
            self.encoder_blocks.append(nn.ModuleList(level_blocks))
            
            # Downsample (except at the last level)
            if level != len(channel_mult) - 1:
                self.downsample_blocks.append(Downsample(current_channels))
            else:
                self.downsample_blocks.append(nn.Identity())
        
        # Middle blocks
        mid_channels = base_channels * channel_mult[-1]
        self.middle = nn.ModuleList([
            ResBlock(
                mid_channels,
                mid_channels,
                time_embed_dim,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(mid_channels, num_heads=num_heads),
            ResBlock(
                mid_channels,
                mid_channels,
                time_embed_dim,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        ])
        
        # Decoder (upsampling path)
        self.upsample_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        for level in reversed(range(len(channel_mult))):
            out_channels_level = base_channels * channel_mult[level]
            
            # Resolution at this level
            resolution = 64 // (2 ** level)
            
            for i in range(num_res_blocks + 1):
                # First ResBlock in each decoder level needs to handle concatenated skip connection
                # So input channels are current_channels + skip_channels
                if i == 0 and level != len(channel_mult) - 1:
                    # Skip connection from encoder
                    in_channels_block = current_channels + out_channels_level
                else:
                    in_channels_block = current_channels
                
                layers = [ResBlock(
                    in_channels_block,
                    out_channels_level,
                    time_embed_dim,
                    dropout=dropout,
                    use_scale_shift_norm=use_scale_shift_norm,
                )]
                
                # Add attention if this resolution is in attention_resolutions
                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(out_channels_level, num_heads=num_heads))
                
                self.decoder_blocks.append(nn.ModuleList(layers))
                current_channels = out_channels_level
            
            # Upsample (except at the first level)
            if level != 0:
                self.upsample_blocks.append(Upsample(current_channels))
            else:
                self.upsample_blocks.append(nn.Identity())
        
        # Output projection
        self.output_norm = GroupNorm32(32, base_channels)
        self.output_conv = nn.Conv2d(base_channels, self.out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the U-Net.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
               This is typically the noisy image x_t
            t: Timestep tensor of shape (batch_size,)

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Input convolution
        h = self.input_conv(x)
        
        # Encoder with skip connections
        skip_connections = []
        
        for level_idx, (level_blocks, downsample) in enumerate(zip(self.encoder_blocks, self.downsample_blocks)):
            # level_blocks contains multiple block groups (one per res_block)
            for block_group in level_blocks:
                # Each block_group is [ResBlock, maybe AttentionBlock]
                for block in block_group:
                    if isinstance(block, ResBlock):
                        h = block(h, t_emb)
                    else:  # AttentionBlock
                        h = block(h)
            
            # Save skip connection for all levels except the last (deepest)
            # The decoder doesn't need a skip at the deepest level
            if level_idx < len(self.encoder_blocks) - 1:
                skip_connections.append(h)
            h = downsample(h)
        
        # Middle blocks
        for block in self.middle:
            if isinstance(block, ResBlock):
                h = block(h, t_emb)
            else:  # AttentionBlock
                h = block(h)
        
        # Decoder with skip connections
        for level_idx in range(len(self.channel_mult)):
            # Process decoder blocks for this level
            for i in range(self.num_res_blocks + 1):
                # Get corresponding decoder blocks
                decoder_block_idx = level_idx * (self.num_res_blocks + 1) + i
                
                # Concatenate with skip connection for first block of each level (except the deepest)
                if i == 0 and level_idx != 0:
                    skip = skip_connections.pop()
                    h = torch.cat([h, skip], dim=1)
                
                for block in self.decoder_blocks[decoder_block_idx]:
                    if isinstance(block, ResBlock):
                        h = block(h, t_emb)
                    else:  # AttentionBlock
                        h = block(h)
            
            # Upsample
            h = self.upsample_blocks[level_idx](h)
        
        # Output projection
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)
        
        return h


def create_model_from_config(config: dict) -> UNet:
    """
    Factory function to create a UNet from a configuration dictionary.
    
    Args:
        config: Dictionary containing model configuration
                Expected to have a 'model' key with the relevant parameters
    
    Returns:
        Instantiated UNet model
    """
    model_config = config['model']
    data_config = config['data']
    
    return UNet(
        in_channels=data_config['channels'],
        out_channels=data_config['channels'],
        base_channels=model_config['base_channels'],
        channel_mult=tuple(model_config['channel_mult']),
        num_res_blocks=model_config['num_res_blocks'],
        attention_resolutions=model_config['attention_resolutions'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout'],
        use_scale_shift_norm=model_config['use_scale_shift_norm'],
    )


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test the model
    print("Testing UNet...")
    
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        num_heads=4,
        dropout=0.1,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    t = torch.rand(batch_size)
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Forward pass successful!")
