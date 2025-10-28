#!/usr/bin/env python3
"""
Test script to validate CBAM integration into Faster R-CNN.

This script tests:
1. CBAM attention modules work correctly
2. CBAM-enhanced backbone can be created
3. Full model with CBAM can be instantiated
4. Forward pass works with sample data

Run this before starting training to ensure everything is set up correctly.
"""

import torch
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*70)
print("🔬 CBAM Integration Validation Script")
print("="*70)

# Test 1: Import attention modules
print("\n[Test 1/5] Testing attention modules import...")
try:
    from DeepDataMiningLearning.detection.attention_modules import (
        CBAM, 
        ChannelAttention, 
        SpatialAttention
    )
    print("✅ Attention modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import attention modules: {e}")
    sys.exit(1)

# Test 2: Test CBAM forward pass
print("\n[Test 2/5] Testing CBAM forward pass...")
try:
    cbam = CBAM(channels=256, reduction=16)
    x = torch.randn(4, 256, 50, 50)
    output = cbam(x)
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    print(f"✅ CBAM forward pass successful: {x.shape} -> {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in cbam.parameters()):,}")
except Exception as e:
    print(f"❌ CBAM forward pass failed: {e}")
    sys.exit(1)

# Test 3: Test CBAM backbone
print("\n[Test 3/5] Testing CBAM-enhanced backbone...")
try:
    from DeepDataMiningLearning.detection.backbone import MyBackboneWithFPN_CBAM
    
    backbone = MyBackboneWithFPN_CBAM(
        model_name='resnet50',  # Use ResNet50 for faster testing
        trainable_layers=0,
        out_channels=256,
        cbam_reduction=16
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 800, 800)
    features = backbone(x)
    
    print(f"✅ CBAM backbone created successfully")
    print(f"   Output feature maps:")
    for k, v in features.items():
        print(f"      {k}: {v.shape}")
    
except Exception as e:
    print(f"❌ CBAM backbone test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test full model creation (without CBAM)
print("\n[Test 4/5] Testing baseline model creation...")
try:
    from DeepDataMiningLearning.detection.models import create_detectionmodel
    
    baseline_model, _, _ = create_detectionmodel(
        modelname='customrcnn_resnet50',  # ResNet50 for faster test
        num_classes=5,
        trainable_layers=0,
        device='cpu'  # Use CPU for testing
    )
    print("✅ Baseline model created successfully")
    
except Exception as e:
    print(f"❌ Baseline model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test full model creation (with CBAM)
print("\n[Test 5/5] Testing CBAM model creation...")
try:
    cbam_model, _, _ = create_detectionmodel(
        modelname='customrcnn_resnet50_cbam',  # With CBAM suffix
        num_classes=5,
        trainable_layers=0,
        device='cpu'
    )
    print("✅ CBAM model created successfully")
    
    # Test inference mode
    cbam_model.eval()
    x = [torch.randn(3, 800, 800), torch.randn(3, 600, 800)]
    
    with torch.no_grad():
        predictions = cbam_model(x)
    
    print(f"✅ Model inference successful")
    print(f"   Predictions for {len(predictions)} images")
    print(f"   Keys: {list(predictions[0].keys())}")
    
except Exception as e:
    print(f"❌ CBAM model creation/inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("🎉 All tests passed! CBAM integration is ready.")
print("="*70)
print("\n📝 Next Steps:")
print("   1. Train baseline model:")
print("      --model 'customrcnn_resnet152'")
print("   2. Train CBAM model:")
print("      --model 'customrcnn_resnet152_cbam'")
print("   3. Compare mAP metrics in your report")
print("\n✨ Good luck with your training!")
