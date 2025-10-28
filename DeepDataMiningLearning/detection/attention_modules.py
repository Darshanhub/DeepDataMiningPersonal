"""
Attention Modules for Object Detection
Implements CBAM (Convolutional Block Attention Module) and related attention mechanisms.

Reference:
    Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018).
    "CBAM: Convolutional block attention module." ECCV 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module from CBAM.
    
    Focuses on "what" is meaningful in the feature map by computing attention
    across the channel dimension using both average and max pooling.
    
    Args:
        channels (int): Number of input channels
        reduction (int): Channel reduction ratio for the MLP bottleneck (default: 16)
    """
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP (implemented as 1x1 convolutions)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]
        Returns:
            Channel attention map [B, C, H, W]
        """
        # Average pool branch
        avg_out = self.mlp(self.avg_pool(x))
        
        # Max pool branch
        max_out = self.mlp(self.max_pool(x))
        
        # Combine and apply sigmoid
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module from CBAM.
    
    Focuses on "where" is an informative part by computing attention
    across the spatial dimension using both average and max pooling.
    
    Args:
        kernel_size (int): Convolution kernel size for spatial attention (default: 7)
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]
        Returns:
            Spatial attention map [B, C, H, W]
        """
        # Average pooling across channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        # Max pooling across channel dimension
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        
        return x * out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Combines Channel Attention and Spatial Attention sequentially to
    refine feature representations for object detection tasks.
    
    Architecture:
        Input → Channel Attention → Spatial Attention → Output
    
    Args:
        channels (int): Number of input channels
        reduction (int): Channel reduction ratio for MLP (default: 16)
        spatial_kernel (int): Kernel size for spatial attention (default: 7)
    
    Example:
        >>> cbam = CBAM(channels=256, reduction=16)
        >>> x = torch.randn(8, 256, 50, 50)
        >>> out = cbam(x)
        >>> print(out.shape)  # torch.Size([8, 256, 50, 50])
    """
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)
        
    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]
        Returns:
            Attention-refined feature map [B, C, H, W]
        """
        # Apply channel attention first
        x = self.channel_attention(x)
        
        # Then apply spatial attention
        x = self.spatial_attention(x)
        
        return x


class CBAMBottleneck(nn.Module):
    """
    ResNet Bottleneck with CBAM attention inserted after conv3.
    
    This is designed to replace standard ResNet bottleneck blocks
    to add attention mechanism without changing the architecture flow.
    
    Args:
        inplanes (int): Number of input channels
        planes (int): Number of bottleneck channels
        stride (int): Stride for conv2 (default: 1)
        downsample (nn.Module): Downsample layer for residual connection (default: None)
        reduction (int): CBAM channel reduction ratio (default: 16)
    """
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(CBAMBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        # CBAM attention module
        self.cbam = CBAM(planes * self.expansion, reduction)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Apply CBAM attention
        out = self.cbam(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


def add_cbam_to_resnet_layer(resnet_layer, channels, reduction=16):
    """
    Utility function to add CBAM modules to specific ResNet layers.
    
    This function wraps each bottleneck block in a ResNet layer with CBAM.
    Useful for retrofitting existing ResNet models with attention.
    
    Args:
        resnet_layer (nn.Sequential): A ResNet layer (e.g., layer3, layer4)
        channels (int): Number of channels in the layer
        reduction (int): CBAM reduction ratio
    
    Returns:
        nn.Sequential: Modified layer with CBAM modules
    
    Example:
        >>> from torchvision.models import resnet50
        >>> model = resnet50(pretrained=True)
        >>> model.layer4 = add_cbam_to_resnet_layer(model.layer4, 2048, reduction=16)
    """
    new_layer = nn.Sequential()
    
    for i, block in enumerate(resnet_layer):
        # Add the original block
        new_layer.add_module(f'block_{i}', block)
        
        # Add CBAM after the block
        cbam = CBAM(channels, reduction=reduction)
        new_layer.add_module(f'cbam_{i}', cbam)
    
    return new_layer


class LightweightCBAM(nn.Module):
    """
    Lightweight version of CBAM with reduced computational cost.
    
    Uses depthwise separable convolutions in spatial attention to
    reduce parameters and computation while maintaining effectiveness.
    
    Args:
        channels (int): Number of input channels
        reduction (int): Channel reduction ratio (default: 16)
    """
    def __init__(self, channels, reduction=16):
        super(LightweightCBAM, self).__init__()
        
        # Channel attention (same as CBAM)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        # Lightweight spatial attention (depthwise separable conv)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=1, groups=2, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Channel attention
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.spatial_conv(spatial_in))
        x = x * spatial_att
        
        return x


# ============================================================================
# Testing and Validation Functions
# ============================================================================

def test_cbam():
    """Test CBAM module with various input sizes."""
    print("=" * 70)
    print("Testing CBAM Module")
    print("=" * 70)
    
    # Test configurations
    test_configs = [
        (8, 256, 50, 50),   # Typical FPN P4 feature map
        (8, 512, 25, 25),   # Typical FPN P5 feature map
        (8, 1024, 13, 13),  # Typical ResNet layer3 output
        (8, 2048, 7, 7),    # Typical ResNet layer4 output
    ]
    
    for batch, channels, height, width in test_configs:
        print(f"\nTest: Input shape [{batch}, {channels}, {height}, {width}]")
        
        # Create module
        cbam = CBAM(channels, reduction=16)
        
        # Create random input
        x = torch.randn(batch, channels, height, width)
        
        # Forward pass
        output = cbam(x)
        
        # Validate output shape
        assert output.shape == x.shape, f"Shape mismatch! {output.shape} != {x.shape}"
        
        # Count parameters
        params = sum(p.numel() for p in cbam.parameters())
        
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Parameters: {params:,}")
        print(f"✓ Forward pass successful")
    
    print("\n" + "=" * 70)
    print("All CBAM tests passed! ✅")
    print("=" * 70)


def test_channel_attention():
    """Test Channel Attention module independently."""
    print("\nTesting Channel Attention Module...")
    
    ca = ChannelAttention(256, reduction=16)
    x = torch.randn(4, 256, 50, 50)
    out = ca(x)
    
    assert out.shape == x.shape
    print(f"✓ Channel Attention: {x.shape} -> {out.shape}")


def test_spatial_attention():
    """Test Spatial Attention module independently."""
    print("\nTesting Spatial Attention Module...")
    
    sa = SpatialAttention(kernel_size=7)
    x = torch.randn(4, 256, 50, 50)
    out = sa(x)
    
    assert out.shape == x.shape
    print(f"✓ Spatial Attention: {x.shape} -> {out.shape}")


def visualize_attention_maps(x, cbam_module):
    """
    Visualize attention maps for debugging/analysis.
    
    Args:
        x: Input tensor [B, C, H, W]
        cbam_module: CBAM module instance
    
    Returns:
        dict: Dictionary containing channel and spatial attention maps
    """
    with torch.no_grad():
        # Channel attention
        avg_pool = cbam_module.channel_attention.avg_pool(x)
        max_pool = cbam_module.channel_attention.max_pool(x)
        avg_out = cbam_module.channel_attention.mlp(avg_pool)
        max_out = cbam_module.channel_attention.mlp(max_pool)
        channel_att = torch.sigmoid(avg_out + max_out)
        
        # Spatial attention (after channel attention)
        x_ch = x * channel_att
        avg_spatial = torch.mean(x_ch, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x_ch, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_att = torch.sigmoid(cbam_module.spatial_attention.conv(spatial_in))
    
    return {
        'channel_attention': channel_att.squeeze().cpu(),
        'spatial_attention': spatial_att.squeeze().cpu(),
        'avg_pool': avg_pool.squeeze().cpu(),
        'max_pool': max_pool.squeeze().cpu()
    }


if __name__ == "__main__":
    # Run all tests
    print("\n🔬 Running CBAM Attention Module Tests\n")
    
    test_channel_attention()
    test_spatial_attention()
    test_cbam()
    
    print("\n" + "=" * 70)
    print("🎉 All tests completed successfully!")
    print("=" * 70)
    print("\nReady to integrate CBAM into your detection model!")
    print("Use: from DeepDataMiningLearning.detection.attention_modules import CBAM")
