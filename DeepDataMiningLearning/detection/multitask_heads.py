"""
Multi-Task Learning Heads for Object Detection

This module implements additional task heads that can be attached to the 
Faster R-CNN backbone for multi-task learning:
    1. Object Detection (existing)
    2. Semantic Segmentation (FCN-style head)
    3. Depth Estimation (regression head)

The shared ResNet152+FPN backbone learns richer representations by
optimizing for multiple related tasks simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from torch import Tensor


class SemanticSegmentationHead(nn.Module):
    """
    Semantic Segmentation Head for Multi-Task Learning.
    
    Uses Feature Pyramid Network outputs to perform pixel-wise classification.
    Architecture inspired by FCN and DeepLabV3.
    
    Args:
        in_channels: Number of input channels from FPN (default: 256)
        num_classes: Number of segmentation classes (5 for Waymo: background + 4 object classes)
        intermediate_channels: Number of channels in intermediate layers (default: 256)
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 5,
        intermediate_channels: int = 256,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Segmentation decoder - processes FPN features
        self.seg_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
        )
        
        self.seg_conv2 = nn.Sequential(
            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
        )
        
        # Final classifier
        self.classifier = nn.Conv2d(intermediate_channels, num_classes, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: Dict[str, Tensor], targets: Optional[List[Dict[str, Tensor]]] = None) -> Dict[str, Tensor]:
        """
        Forward pass for semantic segmentation.
        
        Args:
            features: Dict of feature maps from FPN, e.g., {'0': tensor, '1': tensor, ...}
            targets: Optional list of target dicts with 'seg_masks' (H, W) during training
        
        Returns:
            Dict containing:
                - 'seg_logits': (N, num_classes, H, W) segmentation logits
                - 'loss_segmentation': Segmentation loss (if targets provided)
        """
        # Use the highest resolution FPN feature (stride 4)
        # FPN outputs: '0' (stride 4), '1' (stride 8), '2' (stride 16), '3' (stride 32), 'pool' (stride 64)
        x = features['0']  # Shape: (N, 256, H/4, W/4)
        
        # Apply segmentation decoder
        x = self.seg_conv1(x)
        x = self.seg_conv2(x)
        
        # Generate segmentation logits
        seg_logits = self.classifier(x)  # (N, num_classes, H/4, W/4)
        
        # Upsample to original image size
        seg_logits = F.interpolate(
            seg_logits,
            scale_factor=4,
            mode='bilinear',
            align_corners=False
        )  # (N, num_classes, H, W)
        
        result = {'seg_logits': seg_logits}
        
        # Compute loss if targets are provided
        if targets is not None and len(targets) > 0 and 'seg_masks' in targets[0]:
            # Stack target masks: (N, H, W)
            target_masks = torch.stack([t['seg_masks'] for t in targets])
            
            # Resize targets to match prediction size if needed
            if target_masks.shape[-2:] != seg_logits.shape[-2:]:
                target_masks = F.interpolate(
                    target_masks.unsqueeze(1).float(),
                    size=seg_logits.shape[-2:],
                    mode='nearest'
                ).squeeze(1).long()
            
            # Cross-entropy loss for semantic segmentation
            loss_seg = F.cross_entropy(seg_logits, target_masks, ignore_index=255)
            result['loss_segmentation'] = loss_seg
        
        return result


class DepthEstimationHead(nn.Module):
    """
    Depth Estimation Head for Multi-Task Learning.
    
    Predicts per-pixel depth values to help the network learn 3D structure.
    This improves object detection by providing scale and distance cues.
    
    Args:
        in_channels: Number of input channels from FPN (default: 256)
        intermediate_channels: Number of channels in intermediate layers (default: 128)
        min_depth: Minimum depth value in meters (default: 1.0)
        max_depth: Maximum depth value in meters (default: 80.0)
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        intermediate_channels: int = 128,
        min_depth: float = 1.0,
        max_depth: float = 80.0,
    ):
        super().__init__()
        
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # Depth decoder
        self.depth_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
        )
        
        self.depth_conv2 = nn.Sequential(
            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
        )
        
        # Final depth regressor (output 1 channel for depth)
        self.depth_predictor = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1),
            nn.Sigmoid()  # Normalize to [0, 1], then scale to [min_depth, max_depth]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: Dict[str, Tensor], targets: Optional[List[Dict[str, Tensor]]] = None) -> Dict[str, Tensor]:
        """
        Forward pass for depth estimation.
        
        Args:
            features: Dict of feature maps from FPN
            targets: Optional list of target dicts with 'depth_maps' (H, W) during training
        
        Returns:
            Dict containing:
                - 'depth_pred': (N, 1, H, W) predicted depth maps
                - 'loss_depth': Depth estimation loss (if targets provided)
        """
        # Use highest resolution FPN feature
        x = features['0']  # (N, 256, H/4, W/4)
        
        # Apply depth decoder
        x = self.depth_conv1(x)
        x = self.depth_conv2(x)
        
        # Predict normalized depth [0, 1]
        depth_normalized = self.depth_predictor(x)  # (N, 1, H/4, W/4)
        
        # Scale to actual depth range
        depth_pred = depth_normalized * (self.max_depth - self.min_depth) + self.min_depth
        
        # Upsample to original image size
        depth_pred = F.interpolate(
            depth_pred,
            scale_factor=4,
            mode='bilinear',
            align_corners=False
        )  # (N, 1, H, W)
        
        result = {'depth_pred': depth_pred}
        
        # Compute loss if targets are provided
        if targets is not None and len(targets) > 0 and 'depth_maps' in targets[0]:
            # Stack target depth maps: (N, 1, H, W)
            target_depth = torch.stack([t['depth_maps'] for t in targets]).unsqueeze(1)
            
            # Resize targets if needed
            if target_depth.shape[-2:] != depth_pred.shape[-2:]:
                target_depth = F.interpolate(
                    target_depth,
                    size=depth_pred.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # L1 loss for depth estimation (robust to outliers)
            # Create valid mask (ignore pixels with depth = 0 or > max_depth)
            valid_mask = (target_depth > 0) & (target_depth < self.max_depth)
            
            if valid_mask.sum() > 0:
                loss_depth = F.l1_loss(
                    depth_pred[valid_mask],
                    target_depth[valid_mask]
                )
            else:
                loss_depth = torch.tensor(0.0, device=depth_pred.device)
            
            result['loss_depth'] = loss_depth
        
        return result


class MultiTaskWrapper(nn.Module):
    """
    Wrapper that adds multi-task learning heads to existing Faster R-CNN model.
    
    This wrapper:
        1. Keeps the existing detection pipeline (RPN + RoI Heads)
        2. Adds semantic segmentation head
        3. Adds depth estimation head
        4. Manages multi-task loss weighting
    
    Args:
        detection_model: Existing CustomRCNN model
        num_seg_classes: Number of segmentation classes (default: 5)
        enable_segmentation: Enable segmentation task (default: True)
        enable_depth: Enable depth estimation task (default: True)
        seg_weight: Loss weight for segmentation (default: 1.0)
        depth_weight: Loss weight for depth estimation (default: 0.5)
    """
    
    def __init__(
        self,
        detection_model: nn.Module,
        num_seg_classes: int = 5,
        enable_segmentation: bool = True,
        enable_depth: bool = True,
        seg_weight: float = 1.0,
        depth_weight: float = 0.5,
    ):
        super().__init__()
        
        self.detection_model = detection_model
        self.enable_segmentation = enable_segmentation
        self.enable_depth = enable_depth
        self.seg_weight = seg_weight
        self.depth_weight = depth_weight
        
        # Get FPN output channels
        fpn_out_channels = detection_model.backbone.out_channels  # Should be 256
        
        # Add segmentation head
        if enable_segmentation:
            self.seg_head = SemanticSegmentationHead(
                in_channels=fpn_out_channels,
                num_classes=num_seg_classes
            )
            print(f"✅ [MTL] Semantic Segmentation head added (classes: {num_seg_classes})")
        
        # Add depth estimation head
        if enable_depth:
            self.depth_head = DepthEstimationHead(
                in_channels=fpn_out_channels
            )
            print(f"✅ [MTL] Depth Estimation head added")
        
        print(f"📊 [MTL] Loss weights - Seg: {seg_weight}, Depth: {depth_weight}")
    
    def forward(self, images, targets=None):
        """
        Forward pass with multi-task learning.
        
        During training:
            Returns dict with losses from all tasks
        During inference:
            Returns detection predictions + auxiliary predictions
        """
        # The detection model handles the image conversion internally
        # Just pass through to the detection model first
        
        # 1. Object Detection (existing pipeline)
        # This will handle image preprocessing and return either losses or predictions
        if self.training:
            # Training mode: get detection losses
            detection_output = self.detection_model(images, targets)
        else:
            # Inference mode: get detection predictions  
            detection_output = self.detection_model(images, targets)
        
        # 2. Get FPN features for auxiliary tasks
        # We need to re-extract features from the backbone
        # Convert images to proper format if needed
        if isinstance(images, list):
            # Stack images into a batch tensor
            # Pad to same size (simple version - take max dimensions)
            max_h = max(img.shape[1] for img in images)
            max_w = max(img.shape[2] for img in images)
            
            batch_images = []
            for img in images:
                if img.shape[1:] != (max_h, max_w):
                    # Simple padding to max size
                    padded = torch.zeros(img.shape[0], max_h, max_w, device=img.device, dtype=img.dtype)
                    padded[:, :img.shape[1], :img.shape[2]] = img
                    batch_images.append(padded)
                else:
                    batch_images.append(img)
            
            image_tensor = torch.stack(batch_images)
        else:
            image_tensor = images
        
        # Extract backbone features for auxiliary tasks
        features = self.detection_model.backbone(image_tensor)
        
        # 3. Semantic Segmentation
        seg_output = {}
        if self.enable_segmentation:
            seg_output = self.seg_head(features, targets)
        
        # 4. Depth Estimation
        depth_output = {}
        if self.enable_depth:
            depth_output = self.depth_head(features, targets)
        
        # Combine outputs
        if self.training:
            # Combine all losses
            losses = {}
            
            # Detection losses
            if isinstance(detection_output, dict):
                losses.update(detection_output)
            
            # Segmentation loss (weighted)
            if 'loss_segmentation' in seg_output:
                losses['loss_segmentation'] = seg_output['loss_segmentation'] * self.seg_weight
            
            # Depth loss (weighted)
            if 'loss_depth' in depth_output:
                losses['loss_depth'] = depth_output['loss_depth'] * self.depth_weight
            
            return losses
        else:
            # Return predictions from all tasks
            return {
                'detection': detection_output,
                'segmentation': seg_output.get('seg_logits'),
                'depth': depth_output.get('depth_pred'),
            }


def wrap_model_for_multitask(
    detection_model: nn.Module,
    num_seg_classes: int = 5,
    enable_segmentation: bool = True,
    enable_depth: bool = True,
    seg_weight: float = 1.0,
    depth_weight: float = 0.5,
) -> MultiTaskWrapper:
    """
    Convenience function to wrap a detection model with multi-task learning heads.
    
    Args:
        detection_model: CustomRCNN model
        num_seg_classes: Number of segmentation classes
        enable_segmentation: Whether to enable segmentation task
        enable_depth: Whether to enable depth estimation task
        seg_weight: Loss weight for segmentation
        depth_weight: Loss weight for depth estimation
    
    Returns:
        MultiTaskWrapper model ready for training
    
    Example:
        >>> base_model = CustomRCNN(...)
        >>> mtl_model = wrap_model_for_multitask(base_model, num_seg_classes=5)
        >>> losses = mtl_model(images, targets)
    """
    print("\n" + "="*60)
    print("🔧 [MTL] Wrapping model for Multi-Task Learning")
    print("="*60)
    
    mtl_model = MultiTaskWrapper(
        detection_model=detection_model,
        num_seg_classes=num_seg_classes,
        enable_segmentation=enable_segmentation,
        enable_depth=enable_depth,
        seg_weight=seg_weight,
        depth_weight=depth_weight,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in mtl_model.parameters())
    det_params = sum(p.numel() for p in detection_model.parameters())
    aux_params = total_params - det_params
    
    print(f"\n📊 [MTL] Parameter Summary:")
    print(f"   Detection model: {det_params:,} params")
    print(f"   Auxiliary tasks: {aux_params:,} params")
    print(f"   Total MTL model: {total_params:,} params")
    print(f"   Overhead: {aux_params/det_params*100:.1f}%")
    print("="*60 + "\n")
    
    return mtl_model


if __name__ == "__main__":
    # Test multi-task heads
    print("Testing Multi-Task Learning Heads...")
    
    # Create dummy FPN features
    batch_size = 2
    features = {
        '0': torch.randn(batch_size, 256, 200, 300),  # stride 4
        '1': torch.randn(batch_size, 256, 100, 150),  # stride 8
        '2': torch.randn(batch_size, 256, 50, 75),     # stride 16
        '3': torch.randn(batch_size, 256, 25, 38),     # stride 32
        'pool': torch.randn(batch_size, 256, 13, 19),  # stride 64
    }
    
    # Test segmentation head
    seg_head = SemanticSegmentationHead(in_channels=256, num_classes=5)
    seg_out = seg_head(features)
    print(f"✅ Segmentation output shape: {seg_out['seg_logits'].shape}")
    
    # Test depth head
    depth_head = DepthEstimationHead(in_channels=256)
    depth_out = depth_head(features)
    print(f"✅ Depth output shape: {depth_out['depth_pred'].shape}")
    
    print("\n✅ All tests passed!")
