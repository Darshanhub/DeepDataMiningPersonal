# Colab Verification Instructions

## Step 1: Run Verification Script

Add this cell to your Colab notebook **before** training:

```python
# Verify training setup before starting
!python verify_training_setup.py \
    --data-path /content/gdrive/MyDrive/data/ias_assignment_1/Copy_of_images/unzipped \
    --annotationfile /content/gdrive/MyDrive/data/ias_assignment_1/Copy_of_images/unzipped/Copy_of_annotations.json \
    --model DeepDataMiningLearning/detection/modules/yolov8.yaml \
    --expected-classes "0:Vehicle,1:Pedestrian,2:Cyclist,3:Sign"
```

## What the script verifies:

1. ✅ **File Paths**: Checks if data path and annotation file exist
2. ✅ **COCO Annotations**: Validates annotation structure, class names, and distribution
3. ✅ **data.yaml**: Checks if data.yaml exists and matches COCO annotations
4. ✅ **Dataset Loading**: Tests that dataset loads correctly in dictionary format
5. ✅ **Model Config**: Validates model YAML file
6. ✅ **CUDA/GPU**: Confirms GPU availability

## Expected Output:

If everything is correct, you should see:

```
===============================================================================
  Verification Summary
===============================================================================
✅ All verifications passed!
🚀 Ready to start training!

ℹ️  You can now run your training command:
  python DeepDataMiningLearning/detection/mytrain_yolo.py ...
```

## What to check:

### ✅ Class Names Match
The script will show:
- Classes from COCO annotations (with category IDs)
- Classes from data.yaml (if exists)
- Comparison between expected vs actual classes

**Important**: Make sure class names match exactly:
- COCO annotations: `{1: 'Vehicle', 2: 'Pedestrian', 3: 'Cyclist', 4: 'Sign'}`
- Expected classes: `{0: 'Vehicle', 1: 'Pedestrian', 2: 'Cyclist', 3: 'Sign'}`

Note the **different indexing** (COCO uses 1-4, YOLO uses 0-3)!

### ✅ Annotation Distribution
Check if class distribution is balanced:
- Vehicle: ~77% (expected - most common)
- Pedestrian: ~22% 
- Cyclist: <1%
- Sign: <1%

### ✅ Sample Format
Verify sample output format:
```
Sample keys: ['img', 'cls', 'bboxes', 'batch_idx']
img: torch.Size([3, 640, 640])
cls: torch.Size([N])
bboxes: torch.Size([N, 4])
Bounding boxes are normalized (0-1 range)
```

## If verification fails:

1. **Path errors**: Check Google Drive mount and file paths
2. **Class mismatch**: Verify annotation file has correct classes
3. **Format errors**: Dataset must return dictionaries, not tuples
4. **CUDA not available**: Training will be slow but can proceed

## Step 2: Run Training (only after verification passes)

```python
!python DeepDataMiningLearning/detection/mytrain_yolo.py \
    --model DeepDataMiningLearning/detection/modules/yolov8.yaml \
    --scale s \
    --data-path /content/gdrive/MyDrive/data/ias_assignment_1/Copy_of_images/unzipped \
    --annotationfile /content/gdrive/MyDrive/data/ias_assignment_1/Copy_of_images/unzipped/Copy_of_annotations.json \
    --dataset waymococo \
    --epochs 50 \
    --batch-size 16 \
    --device cuda:0 \
    --output-dir runs/train \
    --expname yolov8s_baseline_v1
```

## Notes:

- **First run will be slow**: Dataset split generation takes time
- **Subsequent runs are faster**: Split files are cached
- **Expect ~22-23 it/s**: On T4 GPU with batch size 16
- **Training time**: ~2-3 hours for 50 epochs
