"""
Verification Script for YOLOv8 Training Setup
Checks dataset, class names, model configuration, and data format before training.
"""

import os
import sys
import torch
import json
import yaml
from pathlib import Path
from pycocotools.coco import COCO

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def print_success(text):
    """Print success message"""
    print(f"✅ {text}")

def print_error(text):
    """Print error message"""
    print(f"❌ {text}")

def print_warning(text):
    """Print warning message"""
    print(f"⚠️  {text}")

def print_info(text):
    """Print info message"""
    print(f"ℹ️  {text}")


def verify_paths(data_path, annotation_file):
    """Verify that data paths exist"""
    print_header("Step 1: Verifying File Paths")
    
    errors = []
    
    # Check data path
    if not os.path.exists(data_path):
        print_error(f"Data path does not exist: {data_path}")
        errors.append("Data path not found")
    else:
        print_success(f"Data path exists: {data_path}")
    
    # Check annotation file
    if not os.path.exists(annotation_file):
        print_error(f"Annotation file does not exist: {annotation_file}")
        errors.append("Annotation file not found")
    else:
        print_success(f"Annotation file exists: {annotation_file}")
        
        # Check file size
        file_size = os.path.getsize(annotation_file) / (1024*1024)  # MB
        print_info(f"Annotation file size: {file_size:.2f} MB")
    
    # Check for data.yaml
    data_yaml_path = os.path.join(data_path, 'data.yaml')
    if os.path.exists(data_yaml_path):
        print_success(f"Found data.yaml: {data_yaml_path}")
    else:
        print_warning(f"data.yaml not found in {data_path}")
        print_info("Will extract class names from COCO annotations instead")
    
    return len(errors) == 0, errors


def verify_coco_annotations(annotation_file, expected_classes=None):
    """Verify COCO annotation file structure and class names"""
    print_header("Step 2: Verifying COCO Annotations")
    
    errors = []
    
    try:
        # Load COCO
        print_info("Loading COCO annotations...")
        coco = COCO(annotation_file)
        
        # Get basic statistics
        num_images = len(coco.imgs)
        num_annotations = len(coco.anns)
        num_categories = len(coco.cats)
        
        print_success(f"COCO file loaded successfully")
        print_info(f"  Images: {num_images:,}")
        print_info(f"  Annotations: {num_annotations:,}")
        print_info(f"  Categories: {num_categories}")
        
        # Check categories
        print("\n📋 Category Information:")
        categories = {}
        for cat_id, cat_info in coco.cats.items():
            categories[cat_id] = cat_info['name']
            print(f"  Category ID {cat_id}: {cat_info['name']}")
        
        # Verify expected classes if provided
        if expected_classes:
            print("\n🔍 Verifying Expected Classes:")
            expected_names = set(expected_classes.values())
            actual_names = set(categories.values())
            
            if expected_names == actual_names:
                print_success("All expected classes found in annotations")
            else:
                missing = expected_names - actual_names
                extra = actual_names - expected_names
                
                if missing:
                    print_error(f"Missing classes in annotations: {missing}")
                    errors.append(f"Missing classes: {missing}")
                
                if extra:
                    print_warning(f"Extra classes in annotations: {extra}")
        
        # Check annotation distribution
        print("\n📊 Annotation Distribution:")
        cat_to_anns = {}
        for ann_id, ann in coco.anns.items():
            cat_id = ann['category_id']
            cat_to_anns[cat_id] = cat_to_anns.get(cat_id, 0) + 1
        
        for cat_id in sorted(cat_to_anns.keys()):
            cat_name = categories.get(cat_id, f"Unknown-{cat_id}")
            count = cat_to_anns[cat_id]
            percentage = (count / num_annotations) * 100
            print(f"  {cat_name} (ID {cat_id}): {count:,} annotations ({percentage:.1f}%)")
        
        # Check for common issues
        print("\n🔍 Checking for Data Quality Issues:")
        
        # Check for empty images
        images_with_anns = set(ann['image_id'] for ann in coco.anns.values())
        images_without_anns = set(coco.imgs.keys()) - images_with_anns
        if images_without_anns:
            print_warning(f"Found {len(images_without_anns)} images without annotations")
            print_info(f"  This is OK for validation set, but unusual for training")
        else:
            print_success("All images have at least one annotation")
        
        # Check for invalid bboxes
        invalid_bboxes = 0
        for ann_id, ann in coco.anns.items():
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                if w <= 0 or h <= 0:
                    invalid_bboxes += 1
        
        if invalid_bboxes > 0:
            print_warning(f"Found {invalid_bboxes} annotations with invalid bounding boxes (w<=0 or h<=0)")
            errors.append(f"{invalid_bboxes} invalid bboxes")
        else:
            print_success("All bounding boxes are valid")
        
        return len(errors) == 0, errors, categories
        
    except Exception as e:
        print_error(f"Failed to load COCO annotations: {str(e)}")
        errors.append(f"COCO loading error: {str(e)}")
        return False, errors, {}


def verify_data_yaml(data_path, coco_categories):
    """Verify data.yaml if it exists, or create a template"""
    print_header("Step 3: Verifying data.yaml Configuration")
    
    data_yaml_path = os.path.join(data_path, 'data.yaml')
    
    if os.path.exists(data_yaml_path):
        try:
            with open(data_yaml_path, 'r') as f:
                data_yaml = yaml.safe_load(f)
            
            print_success(f"data.yaml loaded successfully")
            
            # Check for required fields
            if 'names' not in data_yaml:
                print_error("data.yaml missing 'names' field")
                return False, ["Missing 'names' field"]
            
            # Get class names
            if isinstance(data_yaml['names'], list):
                yaml_classes = {i: name for i, name in enumerate(data_yaml['names'])}
            elif isinstance(data_yaml['names'], dict):
                yaml_classes = data_yaml['names']
            else:
                print_error(f"Invalid 'names' format in data.yaml: {type(data_yaml['names'])}")
                return False, ["Invalid names format"]
            
            print("\n📋 Classes in data.yaml:")
            for class_id, class_name in yaml_classes.items():
                print(f"  Class {class_id}: {class_name}")
            
            # Compare with COCO categories
            print("\n🔍 Comparing with COCO annotations:")
            coco_names = set(coco_categories.values())
            yaml_names = set(yaml_classes.values())
            
            if coco_names == yaml_names:
                print_success("✅ Class names match between data.yaml and COCO annotations!")
            else:
                missing_in_yaml = coco_names - yaml_names
                extra_in_yaml = yaml_names - coco_names
                
                if missing_in_yaml:
                    print_warning(f"Classes in COCO but not in data.yaml: {missing_in_yaml}")
                if extra_in_yaml:
                    print_warning(f"Classes in data.yaml but not in COCO: {extra_in_yaml}")
            
            return True, []
            
        except Exception as e:
            print_error(f"Failed to load data.yaml: {str(e)}")
            return False, [f"data.yaml error: {str(e)}"]
    else:
        print_info("data.yaml not found - will use COCO annotations for class names")
        
        # Show what would be in data.yaml
        print("\n📋 Recommended data.yaml content:")
        print("```yaml")
        print("# Waymo Dataset Configuration")
        print("path: .")
        print("train: .")
        print("val: .")
        print("")
        print("# Classes")
        print("names:")
        for cat_id, cat_name in sorted(coco_categories.items()):
            print(f"  {cat_id}: {cat_name}")
        print("```")
        
        return True, []


def verify_dataset_loading(data_path, annotation_file):
    """Verify that dataset can be loaded correctly"""
    print_header("Step 4: Testing Dataset Loading")
    
    try:
        # Import dataset functions
        from DeepDataMiningLearning.detection.dataset import get_dataset
        
        print_info("Loading training dataset sample...")
        
        # Create minimal args
        class Args:
            def __init__(self):
                self.data_path = data_path
                self.annotationfile = annotation_file
                self.dataset = 'waymococo'
                self.img_size = 640
        
        args = Args()
        
        # Load dataset
        dataset, num_classes = get_dataset(
            'waymococo',
            is_train=True,
            is_val=False,
            args=args,
            output_format='coco'
        )
        
        print_success(f"Dataset loaded successfully")
        print_info(f"  Dataset size: {len(dataset)}")
        print_info(f"  Number of classes: {num_classes}")
        
        # Load one sample
        print_info("\nTesting sample loading...")
        sample = dataset[0]
        
        # Check format
        if isinstance(sample, dict):
            print_success("Sample is in dictionary format (correct for training)")
            print_info(f"  Sample keys: {list(sample.keys())}")
            
            # Check required keys
            required_keys = ['img', 'cls', 'bboxes']
            missing_keys = [k for k in required_keys if k not in sample]
            
            if missing_keys:
                print_error(f"Missing required keys: {missing_keys}")
                return False, [f"Missing keys: {missing_keys}"]
            else:
                print_success("All required keys present: img, cls, bboxes")
            
            # Check tensor shapes
            print_info("\n📐 Tensor Shapes:")
            print(f"  img: {sample['img'].shape} (expected: [3, H, W])")
            print(f"  cls: {sample['cls'].shape} (expected: [N])")
            print(f"  bboxes: {sample['bboxes'].shape} (expected: [N, 4])")
            
            # Check data types
            print_info("\n🔢 Data Types:")
            print(f"  img: {sample['img'].dtype}")
            print(f"  cls: {sample['cls'].dtype}")
            print(f"  bboxes: {sample['bboxes'].dtype}")
            
            # Check value ranges
            print_info("\n📊 Value Ranges:")
            print(f"  img: [{sample['img'].min():.3f}, {sample['img'].max():.3f}]")
            print(f"  cls: {sample['cls'].unique().tolist()} (class IDs)")
            print(f"  bboxes: [{sample['bboxes'].min():.3f}, {sample['bboxes'].max():.3f}]")
            
            # Check if bboxes are normalized
            if sample['bboxes'].max() <= 1.0:
                print_success("Bounding boxes are normalized (0-1 range)")
            else:
                print_warning("Bounding boxes may not be normalized")
            
        else:
            print_error(f"Sample is not a dictionary, got {type(sample)}")
            print_error("Training loop expects dictionary format!")
            return False, ["Wrong sample format"]
        
        return True, []
        
    except Exception as e:
        print_error(f"Dataset loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, [f"Dataset loading error: {str(e)}"]


def verify_model_config(model_yaml_path, num_classes):
    """Verify model configuration file"""
    print_header("Step 5: Verifying Model Configuration")
    
    try:
        # Check if file exists
        if not os.path.exists(model_yaml_path):
            print_error(f"Model YAML not found: {model_yaml_path}")
            return False, ["Model YAML not found"]
        
        print_success(f"Model YAML exists: {model_yaml_path}")
        
        # Load YAML
        with open(model_yaml_path, 'r') as f:
            model_config = yaml.safe_load(f)
        
        print_info(f"Model architecture: {os.path.basename(model_yaml_path)}")
        
        # Check nc (number of classes)
        if 'nc' in model_config:
            yaml_nc = model_config['nc']
            print_info(f"  nc in YAML: {yaml_nc}")
            
            if yaml_nc != num_classes:
                print_warning(f"nc in YAML ({yaml_nc}) will be overridden to {num_classes}")
            else:
                print_success(f"nc matches dataset: {num_classes}")
        
        # Check for custom modules
        if 'backbone' in model_config or 'head' in model_config:
            print_info("Model structure defined in sections")
        
        return True, []
        
    except Exception as e:
        print_error(f"Failed to load model YAML: {str(e)}")
        return False, [f"Model YAML error: {str(e)}"]


def verify_cuda():
    """Verify CUDA availability"""
    print_header("Step 6: Verifying CUDA/GPU Setup")
    
    if torch.cuda.is_available():
        print_success("CUDA is available")
        print_info(f"  CUDA version: {torch.version.cuda}")
        print_info(f"  Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print_info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print_info(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        return True, []
    else:
        print_error("CUDA is not available - training will run on CPU (very slow!)")
        return False, ["CUDA not available"]


def main():
    """Main verification function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify YOLOv8 training setup')
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--annotationfile', type=str, required=True, help='Path to COCO annotation JSON')
    parser.add_argument('--model', type=str, required=True, help='Path to model YAML config')
    parser.add_argument('--expected-classes', type=str, default='0:Vehicle,1:Pedestrian,2:Cyclist,3:Sign',
                       help='Expected class mapping (format: "0:Vehicle,1:Pedestrian,...")')
    
    args = parser.parse_args()
    
    # Parse expected classes
    expected_classes = {}
    for item in args.expected_classes.split(','):
        class_id, class_name = item.split(':')
        expected_classes[int(class_id)] = class_name
    
    print("\n" + "🔍 YOLOv8 Training Setup Verification" + "\n")
    print(f"Data Path: {args.data_path}")
    print(f"Annotation File: {args.annotationfile}")
    print(f"Model Config: {args.model}")
    print(f"Expected Classes: {expected_classes}")
    
    all_passed = True
    all_errors = []
    
    # Run all verification steps
    passed, errors = verify_paths(args.data_path, args.annotationfile)
    all_passed = all_passed and passed
    all_errors.extend(errors)
    
    if passed:  # Only continue if paths exist
        passed, errors, coco_categories = verify_coco_annotations(args.annotationfile, expected_classes)
        all_passed = all_passed and passed
        all_errors.extend(errors)
        
        passed, errors = verify_data_yaml(args.data_path, coco_categories)
        all_passed = all_passed and passed
        all_errors.extend(errors)
        
        passed, errors = verify_dataset_loading(args.data_path, args.annotationfile)
        all_passed = all_passed and passed
        all_errors.extend(errors)
        
        num_classes = len(coco_categories)
        passed, errors = verify_model_config(args.model, num_classes)
        all_passed = all_passed and passed
        all_errors.extend(errors)
    
    passed, errors = verify_cuda()
    if not passed:
        print_warning("CUDA not available - training will be slow but can proceed")
    
    # Final summary
    print_header("Verification Summary")
    
    if all_passed:
        print_success("✅ All verifications passed!")
        print_success("🚀 Ready to start training!")
        print_info("\nYou can now run your training command:")
        print_info("  python DeepDataMiningLearning/detection/mytrain_yolo.py ...")
        return 0
    else:
        print_error("❌ Some verifications failed")
        print_error("\nErrors found:")
        for i, error in enumerate(all_errors, 1):
            print(f"  {i}. {error}")
        print_error("\n⚠️  Please fix these issues before training!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
