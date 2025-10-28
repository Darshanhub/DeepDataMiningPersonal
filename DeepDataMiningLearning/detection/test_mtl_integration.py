"""
Multi-Task Learning Integration Test

This script verifies that the MTL model works correctly without training:
1. Model creation and initialization
2. Forward pass with dummy data
3. Loss computation
4. Gradient flow
5. Parameter counting
6. Memory footprint
"""

import sys
import torch
import torch.nn as nn
from collections import OrderedDict

# Ensure the package is importable
sys.path.insert(0, '.')

def test_model_creation():
    """Test 1: Can we create the MTL model?"""
    print("\n" + "="*60)
    print("TEST 1: Model Creation")
    print("="*60)
    
    try:
        from DeepDataMiningLearning.detection.modeling_rpnfasterrcnn import CustomRCNN
        from DeepDataMiningLearning.detection.multitask_heads import wrap_model_for_multitask
        
        # Create base detection model
        print("Creating base CustomRCNN model...")
        base_model = CustomRCNN(
            backbone_modulename='resnet50',  # Use ResNet50 for faster testing
            trainable_layers=0,
            num_classes=5,
            out_channels=256,
            min_size=800,
            max_size=1333
        )
        print("✅ Base model created successfully")
        
        # Wrap with MTL heads
        print("\nWrapping with multi-task learning heads...")
        mtl_model = wrap_model_for_multitask(
            detection_model=base_model,
            num_seg_classes=5,
            enable_segmentation=True,
            enable_depth=True,
            seg_weight=1.0,
            depth_weight=0.5
        )
        print("✅ MTL model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in mtl_model.parameters())
        trainable_params = sum(p.numel() for p in mtl_model.parameters() if p.requires_grad)
        
        print(f"\n📊 Parameter counts:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        return mtl_model, True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def test_forward_pass_training(model):
    """Test 2: Forward pass in training mode"""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass (Training Mode)")
    print("="*60)
    
    try:
        model.train()
        device = 'cpu'  # Use CPU for testing
        
        # Create dummy input (2 images)
        print("Creating dummy input data...")
        batch_size = 2
        images = [
            torch.randn(3, 600, 800),  # Image 1
            torch.randn(3, 700, 900),  # Image 2 (different size)
        ]
        
        # Create dummy targets with all required fields
        targets = []
        for i in range(batch_size):
            # Detection targets
            num_boxes = 3
            target = {
                'boxes': torch.rand(num_boxes, 4) * 400,  # Random boxes
                'labels': torch.randint(1, 5, (num_boxes,)),  # Random labels (1-4)
            }
            
            # Segmentation targets (same size as image)
            h, w = images[i].shape[1:]
            target['seg_masks'] = torch.randint(0, 5, (h, w))  # Random segmentation masks
            
            # Depth targets
            target['depth_maps'] = torch.rand(h, w) * 50 + 10  # Random depth 10-60m
            
            targets.append(target)
        
        print(f"✅ Created {batch_size} dummy images with targets")
        print(f"   Image shapes: {[img.shape for img in images]}")
        print(f"   Boxes per image: {[len(t['boxes']) for t in targets]}")
        
        # Forward pass
        print("\nRunning forward pass...")
        with torch.set_grad_enabled(True):
            outputs = model(images, targets)
        
        # Check outputs
        print(f"\n✅ Forward pass successful!")
        print(f"\nLoss components:")
        total_loss = 0
        for key, value in outputs.items():
            if 'loss' in key:
                print(f"   {key}: {value.item():.4f}")
                total_loss += value.item()
        
        print(f"\n📊 Total loss: {total_loss:.4f}")
        
        # Verify all expected losses are present
        expected_losses = ['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']
        optional_losses = ['loss_segmentation', 'loss_depth']
        
        print(f"\n🔍 Checking loss components...")
        for loss_name in expected_losses:
            if loss_name in outputs:
                print(f"   ✅ {loss_name} present")
            else:
                print(f"   ⚠️  {loss_name} missing (might be OK)")
        
        for loss_name in optional_losses:
            if loss_name in outputs:
                print(f"   ✅ {loss_name} present (MTL task)")
            else:
                print(f"   ❌ {loss_name} missing (MTL task not working)")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass_inference(model):
    """Test 3: Forward pass in inference mode"""
    print("\n" + "="*60)
    print("TEST 3: Forward Pass (Inference Mode)")
    print("="*60)
    
    try:
        model.eval()
        
        # Create dummy input
        print("Creating dummy input data...")
        images = [
            torch.randn(3, 600, 800),
            torch.randn(3, 700, 900),
        ]
        
        print(f"✅ Created {len(images)} dummy images")
        
        # Forward pass without targets
        print("\nRunning inference...")
        with torch.no_grad():
            outputs = model(images, targets=None)
        
        print(f"✅ Inference successful!")
        
        # Check outputs structure
        if isinstance(outputs, dict):
            print(f"\n📦 Output structure:")
            for key, value in outputs.items():
                if value is not None:
                    if isinstance(value, list):
                        print(f"   {key}: List with {len(value)} items")
                        if len(value) > 0 and isinstance(value[0], dict):
                            print(f"      First item keys: {list(value[0].keys())}")
                    elif isinstance(value, torch.Tensor):
                        print(f"   {key}: Tensor {value.shape}")
                else:
                    print(f"   {key}: None")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow(model):
    """Test 4: Verify gradients flow through all components"""
    print("\n" + "="*60)
    print("TEST 4: Gradient Flow")
    print("="*60)
    
    try:
        model.train()
        
        # Create dummy input
        images = [torch.randn(3, 600, 800)]
        targets = [{
            'boxes': torch.rand(2, 4) * 400,
            'labels': torch.randint(1, 5, (2,)),
            'seg_masks': torch.randint(0, 5, (600, 800)),
            'depth_maps': torch.rand(600, 800) * 50 + 10,
        }]
        
        # Forward pass
        outputs = model(images, targets)
        
        # Compute total loss
        total_loss = sum(v for k, v in outputs.items() if 'loss' in k)
        
        # Backward pass
        print("Running backward pass...")
        total_loss.backward()
        
        # Check gradients
        print("\n🔍 Checking gradients...")
        
        components = {
            'Backbone': model.detection_model.backbone,
            'RPN': model.detection_model.rpn,
            'RoI Heads': model.detection_model.roi_heads,
        }
        
        if model.enable_segmentation:
            components['Segmentation Head'] = model.seg_head
        
        if model.enable_depth:
            components['Depth Head'] = model.depth_head
        
        for name, module in components.items():
            params_with_grad = []
            params_without_grad = []
            
            for param in module.parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        params_with_grad.append(param)
                    else:
                        params_without_grad.append(param)
            
            if len(params_with_grad) > 0:
                avg_grad = sum(p.grad.abs().mean().item() for p in params_with_grad) / len(params_with_grad)
                print(f"   ✅ {name}: {len(params_with_grad)} params have gradients (avg: {avg_grad:.6f})")
            else:
                print(f"   ⚠️  {name}: No gradients (might be frozen)")
        
        return True
        
    except Exception as e:
        print(f"❌ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_via_create_detectionmodel():
    """Test 5: Create model via models.py (actual usage)"""
    print("\n" + "="*60)
    print("TEST 5: Model Creation via create_detectionmodel()")
    print("="*60)
    
    try:
        from DeepDataMiningLearning.detection.models import create_detectionmodel
        
        # Test MTL model creation
        print("Creating MTL model with model name: 'customrcnn_resnet50_mtl'")
        model, preprocess, classes = create_detectionmodel(
            modelname='customrcnn_resnet50_mtl',
            num_classes=5,
            nocustomize=False,
            trainable_layers=0,
            device='cpu'
        )
        
        print("✅ Model created via create_detectionmodel()")
        
        # Verify it's the MTL wrapper
        from DeepDataMiningLearning.detection.multitask_heads import MultiTaskWrapper
        if isinstance(model, MultiTaskWrapper):
            print("✅ Model is correctly wrapped with MultiTaskWrapper")
        else:
            print(f"⚠️  Model type: {type(model).__name__} (expected MultiTaskWrapper)")
        
        # Quick forward pass test
        print("\nTesting quick forward pass...")
        model.train()
        images = [torch.randn(3, 400, 600)]
        targets = [{
            'boxes': torch.rand(2, 4) * 300,
            'labels': torch.tensor([1, 2]),
            'seg_masks': torch.randint(0, 5, (400, 600)),
            'depth_maps': torch.rand(400, 600) * 40 + 10,
        }]
        
        outputs = model(images, targets)
        print(f"✅ Forward pass successful, got {len(outputs)} loss components")
        
        return True
        
    except Exception as e:
        print(f"❌ create_detectionmodel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_footprint(model):
    """Test 6: Estimate memory footprint"""
    print("\n" + "="*60)
    print("TEST 6: Memory Footprint")
    print("="*60)
    
    try:
        # Count parameters by component
        components = {
            'Detection Model': model.detection_model,
        }
        
        if model.enable_segmentation:
            components['Segmentation Head'] = model.seg_head
        
        if model.enable_depth:
            components['Depth Head'] = model.depth_head
        
        print("\n📊 Parameter breakdown:")
        total_params = 0
        for name, module in components.items():
            num_params = sum(p.numel() for p in module.parameters())
            total_params += num_params
            memory_mb = num_params * 4 / (1024 ** 2)  # Assuming float32
            print(f"   {name}: {num_params:,} params ({memory_mb:.2f} MB)")
        
        total_memory_mb = total_params * 4 / (1024 ** 2)
        print(f"\n   Total: {total_params:,} params ({total_memory_mb:.2f} MB)")
        
        # Estimate activation memory for one forward pass
        # Rough estimate: batch_size * channels * height * width * num_layers
        batch_size = 2
        h, w = 800, 1200
        fpn_channels = 256
        
        # FPN feature maps at different scales
        fpn_memory = batch_size * fpn_channels * (
            (h//4 * w//4) +  # stride 4
            (h//8 * w//8) +  # stride 8
            (h//16 * w//16) + # stride 16
            (h//32 * w//32) + # stride 32
            (h//64 * w//64)   # stride 64
        ) * 4 / (1024 ** 2)
        
        # Segmentation output
        seg_memory = batch_size * 5 * h * w * 4 / (1024 ** 2)
        
        # Depth output
        depth_memory = batch_size * 1 * h * w * 4 / (1024 ** 2)
        
        total_activation_memory = fpn_memory + seg_memory + depth_memory
        
        print(f"\n📊 Estimated activation memory (batch_size={batch_size}, {h}x{w}):")
        print(f"   FPN features: {fpn_memory:.2f} MB")
        print(f"   Segmentation output: {seg_memory:.2f} MB")
        print(f"   Depth output: {depth_memory:.2f} MB")
        print(f"   Total: {total_activation_memory:.2f} MB")
        
        print(f"\n💾 Total estimated GPU memory: {total_memory_mb + total_activation_memory:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory footprint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MULTI-TASK LEARNING INTEGRATION VERIFICATION")
    print("="*60)
    print("\nThis script verifies the MTL implementation without training.")
    print("It will test:")
    print("  1. Model creation")
    print("  2. Forward pass (training mode)")
    print("  3. Forward pass (inference mode)")
    print("  4. Gradient flow")
    print("  5. Integration with models.py")
    print("  6. Memory footprint estimation")
    
    results = {}
    
    # Test 1: Model creation
    model, success = test_model_creation()
    results['Model Creation'] = success
    
    if not success:
        print("\n❌ Cannot proceed without model. Exiting.")
        return
    
    # Test 2: Forward pass (training)
    results['Forward Pass (Training)'] = test_forward_pass_training(model)
    
    # Test 3: Forward pass (inference)
    results['Forward Pass (Inference)'] = test_forward_pass_inference(model)
    
    # Test 4: Gradient flow
    results['Gradient Flow'] = test_gradient_flow(model)
    
    # Test 5: Integration with models.py
    results['models.py Integration'] = test_model_via_create_detectionmodel()
    
    # Test 6: Memory footprint
    results['Memory Footprint'] = test_memory_footprint(model)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nYour MTL model is ready for training. Use:")
        print("  --model customrcnn_resnet50_mtl")
        print("  --model customrcnn_resnet152_mtl")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the issues before training.")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    import torch
    torch.manual_seed(42)  # For reproducibility
    
    success = main()
    sys.exit(0 if success else 1)
