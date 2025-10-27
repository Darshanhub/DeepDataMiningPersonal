#!/usr/bin/env python3
"""
Quick verification script to check all CBAM + Deformable Conv components.
Run this to verify all changes are in place before training.
"""

import sys
import traceback

def check_file_exists():
    """Check if all required files exist."""
    from pathlib import Path
    
    required_files = [
        "DeepDataMiningLearning/detection/modules/attention.py",
        "DeepDataMiningLearning/detection/modules/block.py",
        "DeepDataMiningLearning/detection/modules/yolomodels.py",
        "DeepDataMiningLearning/detection/modules/yolov8.yaml",
        "DeepDataMiningLearning/detection/modules/yolov8_cbam_deform.yaml",
    ]
    
    print("=" * 60)
    print("FILE EXISTENCE CHECK")
    print("=" * 60)
    
    all_exist = True
    for file_path in required_files:
        p = Path(file_path)
        status = "✓" if p.exists() else "✗"
        print(f"{status} {file_path}")
        if not p.exists():
            all_exist = False
    
    print()
    return all_exist


def check_imports():
    """Check if all modules can be imported."""
    print("=" * 60)
    print("IMPORT CHECK")
    print("=" * 60)
    
    try:
        print("Importing attention module...", end=" ")
        from DeepDataMiningLearning.detection.modules.attention import CBAM, ChannelAttention, SpatialAttention
        print("✓")
        
        print("Importing block module (C2fCBAM, DeformConv)...", end=" ")
        from DeepDataMiningLearning.detection.modules.block import C2fCBAM, DeformConv
        print("✓")
        
        print("Importing yolomodels (checking exports)...", end=" ")
        from DeepDataMiningLearning.detection.modules.yolomodels import YoloDetectionModel
        print("✓")
        
        print("\nAll imports successful!")
        print()
        return True
        
    except Exception as e:
        print(f"✗ FAILED")
        print(f"\nError: {e}")
        traceback.print_exc()
        print()
        return False


def check_instantiation():
    """Try to instantiate the modules."""
    print("=" * 60)
    print("INSTANTIATION CHECK")
    print("=" * 60)
    
    try:
        import torch
        from DeepDataMiningLearning.detection.modules.attention import CBAM
        from DeepDataMiningLearning.detection.modules.block import C2fCBAM, DeformConv
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print()
        
        # Test CBAM
        print("Testing CBAM(64)...", end=" ")
        cbam = CBAM(64)
        x = torch.randn(1, 64, 32, 32)
        y = cbam(x)
        print(f"✓ Input: {x.shape} → Output: {y.shape}")
        
        # Test C2fCBAM
        print("Testing C2fCBAM(64, 64)...", end=" ")
        c2f_cbam = C2fCBAM(64, 64)
        x = torch.randn(1, 64, 32, 32)
        y = c2f_cbam(x)
        print(f"✓ Input: {x.shape} → Output: {y.shape}")
        
        # Test DeformConv
        print("Testing DeformConv(64, 128, 3, 2)...", end=" ")
        deform = DeformConv(64, 128, 3, 2)
        x = torch.randn(1, 64, 64, 64)
        y = deform(x)
        print(f"✓ Input: {x.shape} → Output: {y.shape}")
        
        print("\nAll modules instantiated successfully!")
        print()
        return True
        
    except Exception as e:
        print(f"✗ FAILED")
        print(f"\nError: {e}")
        traceback.print_exc()
        print()
        return False


def check_yaml_syntax():
    """Check YAML files for syntax errors."""
    print("=" * 60)
    print("YAML SYNTAX CHECK")
    print("=" * 60)
    
    import yaml
    
    yaml_files = [
        "DeepDataMiningLearning/detection/modules/yolov8.yaml",
        "DeepDataMiningLearning/detection/modules/yolov8_cbam_deform.yaml",
    ]
    
    all_valid = True
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
            print(f"✓ {yaml_file}")
            print(f"  - nc: {data.get('nc', 'N/A')}")
            print(f"  - scales: {list(data.get('scales', {}).keys())}")
            print(f"  - backbone layers: {len(data.get('backbone', []))}")
            print(f"  - head layers: {len(data.get('head', []))}")
        except Exception as e:
            print(f"✗ {yaml_file}")
            print(f"  Error: {e}")
            all_valid = False
    
    print()
    return all_valid


def check_modifications():
    """Check specific code modifications are present."""
    print("=" * 60)
    print("CODE MODIFICATION CHECK")
    print("=" * 60)
    
    checks = []
    
    # Check block.py for C2fCBAM
    with open("DeepDataMiningLearning/detection/modules/block.py", 'r') as f:
        block_content = f.read()
        
    checks.append(("C2fCBAM class in block.py", "class C2fCBAM" in block_content))
    checks.append(("DeformConv class in block.py", "class DeformConv" in block_content))
    checks.append(("CBAM import in block.py", "from DeepDataMiningLearning.detection.modules.attention import CBAM" in block_content))
    checks.append(("DeformConv2d import in block.py", "from torchvision.ops import DeformConv2d" in block_content))
    
    # Check yolomodels.py exports
    with open("DeepDataMiningLearning/detection/modules/yolomodels.py", 'r') as f:
        yolo_content = f.read()
        
    checks.append(("C2fCBAM exported in yolomodels.py", "C2fCBAM" in yolo_content))
    checks.append(("DeformConv exported in yolomodels.py", "DeformConv" in yolo_content))
    
    # Check YAML for modifications
    with open("DeepDataMiningLearning/detection/modules/yolov8_cbam_deform.yaml", 'r') as f:
        yaml_content = f.read()
        
    checks.append(("C2fCBAM in YAML", "C2fCBAM" in yaml_content))
    checks.append(("DeformConv in YAML", "DeformConv" in yaml_content))
    cbam_count = yaml_content.count("C2fCBAM")
    deform_count = yaml_content.count("DeformConv")
    checks.append((f"4x C2fCBAM in YAML (found {cbam_count})", cbam_count == 4))
    checks.append((f"2x DeformConv in YAML (found {deform_count})", deform_count == 2))
    
    all_passed = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    print()
    return all_passed


def main():
    """Run all verification checks."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  YOLOv8 CBAM + Deformable Conv Verification Script  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    results = []
    
    # Run all checks
    results.append(("File Existence", check_file_exists()))
    results.append(("Code Modifications", check_modifications()))
    results.append(("YAML Syntax", check_yaml_syntax()))
    results.append(("Python Imports", check_imports()))
    results.append(("Module Instantiation", check_instantiation()))
    
    # Summary
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for check_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status.ljust(8)} {check_name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ ALL CHECKS PASSED! Ready to train.")
        print("\nNext steps:")
        print("1. Prepare your dataset in COCO format")
        print("2. Review TRAINING_GUIDE.md for commands")
        print("3. Train baseline: yolov8.yaml")
        print("4. Train modified: yolov8_cbam_deform.yaml")
        print("5. Compare mAP results\n")
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED. Review errors above.")
        print("Common fixes:")
        print("- Install dependencies: pip install torch torchvision")
        print("- Install repo: pip install -e .")
        print("- Check file paths are correct\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
