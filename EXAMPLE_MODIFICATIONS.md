# Practical Examples of Model Architecture Modifications

This document provides concrete, ready-to-implement modifications for your object detection project.

---

## Example 1: Enhanced Faster R-CNN with Deeper Backbone

### Modification Description
Replace ResNet50 with ResNet152 and increase trainable layers for better feature extraction.

### Code Changes

**File**: `detection/backbone.py` (no changes needed - already supports it)

**Training Command**:
```bash
python DeepDataMiningLearning/detection/mytrain.py \
    --model customrcnn_resnet152 \
    --data-path /path/to/waymo_subset_coco_4000step5/ \
    --annotationfile /path/to/waymo_subset_coco_4000step5/annotations.json \
    --dataset waymococo \
    --batch-size 4 \
    --epochs 50 \
    --lr 0.001 \
    --trainable 3 \
    --output-dir ./outputs/exp1_resnet152 \
    --device cuda:0
```

### Expected Results
- **Baseline (ResNet50)**: mAP@0.5:0.95 ≈ 0.30-0.35
- **Modified (ResNet152)**: mAP@0.5:0.95 ≈ 0.33-0.38 (+2-3%)
- **Trade-off**: 1.5x more parameters, 30% slower inference

---

## Example 2: Increase FPN Capacity

### Modification Description
Increase FPN output channels from 256 to 512 for richer feature representation.

### Code Changes

**File**: `detection/backbone.py`

**Before** (Line ~8):
```python
def __init__(
    self,
    model_name: str,
    trainable_layers: int,
    out_channels: int = 256,  # Current value
    ...
):
```

**After**:
```python
def __init__(
    self,
    model_name: str,
    trainable_layers: int,
    out_channels: int = 512,  # Increased to 512
    ...
):
```

**OR** when calling model creation in `detection/models.py` line ~225:

**Before**:
```python
model=CustomRCNN(
    backbone_modulename=backbonename,
    trainable_layers=trainable_layers,
    num_classes=num_classes,
    out_channels=256,  # Change this
    min_size=800,
    max_size=1333
)
```

**After**:
```python
model=CustomRCNN(
    backbone_modulename=backbonename,
    trainable_layers=trainable_layers,
    num_classes=num_classes,
    out_channels=512,  # Increased capacity
    min_size=800,
    max_size=1333
)
```

### Training Command
```bash
python DeepDataMiningLearning/detection/mytrain.py \
    --model customrcnn_resnet50 \
    --data-path /path/to/waymo_subset_coco_4000step5/ \
    --annotationfile /path/to/waymo_subset_coco_4000step5/annotations.json \
    --dataset waymococo \
    --batch-size 6 \
    --epochs 50 \
    --lr 0.001 \
    --trainable 2 \
    --output-dir ./outputs/exp2_fpn512 \
    --device cuda:0
```

### Expected Results
- **Baseline (256 channels)**: mAP@0.5:0.95 ≈ 0.30-0.35
- **Modified (512 channels)**: mAP@0.5:0.95 ≈ 0.32-0.37 (+1-2%)
- **Trade-off**: 2x FPN parameters, 15% slower

---

## Example 3: More RPN Anchors for Small Objects

### Modification Description
Add more anchor scales and aspect ratios to detect small objects better (pedestrians, cyclists).

### Code Changes

**File**: `detection/modeling_rpnfasterrcnn.py`

**Before** (Line ~30-35):
```python
class AnchorGenerator(nn.Module):
    def __init__(
        self,
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    ):
```

**After**:
```python
class AnchorGenerator(nn.Module):
    def __init__(
        self,
        sizes=((32, 64, 128, 256, 512),),  # Added smaller anchors
        aspect_ratios=((0.5, 0.75, 1.0, 1.5, 2.0),),  # More aspect ratios
    ):
```

**Also modify** in `CustomRCNN` class (around line 1200):

**Before**:
```python
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)
```

**After**:
```python
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 0.75, 1.0, 1.5, 2.0),)  # More ratios
)
```

### Training Command
```bash
python DeepDataMiningLearning/detection/mytrain.py \
    --model customrcnn_resnet50 \
    --data-path /path/to/waymo_subset_coco_4000step5/ \
    --annotationfile /path/to/waymo_subset_coco_4000step5/annotations.json \
    --dataset waymococo \
    --batch-size 8 \
    --epochs 50 \
    --output-dir ./outputs/exp3_more_anchors \
    --device cuda:0
```

### Expected Results
- **Improvement on small objects**: +3-5% AP for pedestrians and cyclists
- **Overall mAP**: +1-2%
- **Trade-off**: Minimal computation increase

---

## Example 4: YOLOv8 with Deeper Neck

### Modification Description
Increase C2f blocks in YOLO neck from 3 to 6 for better multi-scale fusion.

### Code Changes

**File**: `detection/modules/yolov8.yaml`

**Before** (Lines showing neck section):
```yaml
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 15  # Current: 3 blocks
  
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 18 (P3/8-small)  # Current: 3 blocks
```

**After**:
```yaml
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 6, C2f, [512]]  # 15  # Changed to 6 blocks
  
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 6, C2f, [256]]  # 18 (P3/8-small)  # Changed to 6 blocks
```

### Training Command
```bash
python DeepDataMiningLearning/detection/mytrain_yolo.py \
    --model yolov8 \
    --scale m \
    --data-path /path/to/waymo_subset_coco_4000step5/ \
    --annotationfile /path/to/waymo_subset_coco_4000step5/annotations.json \
    --dataset waymococo \
    --batch-size 16 \
    --epochs 100 \
    --lr 0.01 \
    --output-dir ./outputs/exp4_yolov8_deeper_neck \
    --device cuda:0
```

### Expected Results
- **Baseline YOLOv8m**: mAP@0.5:0.95 ≈ 0.35-0.40
- **Deeper Neck**: mAP@0.5:0.95 ≈ 0.37-0.42 (+2%)
- **Trade-off**: 20% more parameters, 10% slower

---

## Example 5: Add Attention to Detection Head

### Modification Description
Add spatial attention module to YOLO detection head for better feature focus.

### Code Changes

**Step 1**: Create attention module

**File**: Create new file `detection/modules/attention.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    """Spatial attention module for feature enhancement."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_attn = self.conv1(x_cat)
        return x * self.sigmoid(x_attn)

class ChannelAttention(nn.Module):
    """Channel attention module."""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
```

**Step 2**: Modify detection head

**File**: `detection/modules/head.py`

Add import at top:
```python
from DeepDataMiningLearning.detection.modules.attention import CBAM
```

**Modify Detect class** (around line 30-50):

**Before**:
```python
class Detect(nn.Module):
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
```

**After**:
```python
class Detect(nn.Module):
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        
        # Add attention modules
        self.attention = nn.ModuleList(CBAM(x) for x in ch)
        
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
```

**Modify forward method** (around line 80):

**Before**:
```python
def forward(self, x):
    shape = x[0].shape
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
```

**After**:
```python
def forward(self, x):
    shape = x[0].shape
    for i in range(self.nl):
        # Apply attention before detection
        x[i] = self.attention[i](x[i])
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
```

### Training Command
```bash
python DeepDataMiningLearning/detection/mytrain_yolo.py \
    --model yolov8 \
    --scale s \
    --data-path /path/to/waymo_subset_coco_4000step5/ \
    --annotationfile /path/to/waymo_subset_coco_4000step5/annotations.json \
    --dataset waymococo \
    --batch-size 16 \
    --epochs 100 \
    --output-dir ./outputs/exp5_yolo_attention \
    --device cuda:0
```

### Expected Results
- **Baseline YOLOv8s**: mAP@0.5:0.95 ≈ 0.32-0.37
- **With CBAM Attention**: mAP@0.5:0.95 ≈ 0.35-0.40 (+2-3%)
- **Trade-off**: +5% parameters, +8% compute

---

## Example 6: Multi-Scale Training Strategy

### Modification Description
Train with multiple input resolutions to improve detection at different scales.

### Code Changes

**File**: `detection/mytrain.py`

**Add multi-scale transform** (around line 400-450 in main function):

```python
# After dataset creation
if args.multi_scale:
    # Multi-scale training sizes
    img_sizes = [480, 544, 608, 672, 736, 800]
    
    # Create datasets with different sizes
    datasets = []
    for img_size in img_sizes:
        ds, _ = get_dataset(args.dataset, "train", args, img_size=img_size)
        datasets.append(ds)
    
    # Combine datasets
    from torch.utils.data import ConcatDataset
    dataset = ConcatDataset(datasets)
```

**Add argument**:
```python
parser.add_argument("--multi-scale", action="store_true", help="use multi-scale training")
```

### Training Command
```bash
python DeepDataMiningLearning/detection/mytrain.py \
    --model customrcnn_resnet50 \
    --data-path /path/to/waymo_subset_coco_4000step5/ \
    --annotationfile /path/to/waymo_subset_coco_4000step5/annotations.json \
    --dataset waymococo \
    --multi-scale \
    --batch-size 4 \
    --epochs 50 \
    --output-dir ./outputs/exp6_multiscale \
    --device cuda:0
```

### Expected Results
- **Single-scale**: mAP@0.5:0.95 ≈ 0.30-0.35
- **Multi-scale**: mAP@0.5:0.95 ≈ 0.33-0.38 (+2-3%)
- **Benefit**: Better generalization across scales

---

## Evaluation Script

### Create evaluation script

**File**: Create `evaluate_models.sh`:

```bash
#!/bin/bash

# Waymo dataset path
DATA_PATH="/path/to/waymo_subset_coco_4000step5/"
ANNOTATION="${DATA_PATH}annotations.json"

# Evaluate all experiments
echo "=== Evaluating Experiment 1: ResNet152 ==="
python DeepDataMiningLearning/detection/mytrain.py \
    --model customrcnn_resnet152 \
    --data-path $DATA_PATH \
    --annotationfile $ANNOTATION \
    --dataset waymococo \
    --resume ./outputs/exp1_resnet152/checkpoint.pth \
    --test-only \
    --output-dir ./eval_results/exp1

echo "=== Evaluating Experiment 2: FPN 512 ==="
python DeepDataMiningLearning/detection/mytrain.py \
    --model customrcnn_resnet50 \
    --data-path $DATA_PATH \
    --annotationfile $ANNOTATION \
    --dataset waymococo \
    --resume ./outputs/exp2_fpn512/checkpoint.pth \
    --test-only \
    --output-dir ./eval_results/exp2

echo "=== Evaluating Experiment 3: More Anchors ==="
python DeepDataMiningLearning/detection/mytrain.py \
    --model customrcnn_resnet50 \
    --data-path $DATA_PATH \
    --annotationfile $ANNOTATION \
    --dataset waymococo \
    --resume ./outputs/exp3_more_anchors/checkpoint.pth \
    --test-only \
    --output-dir ./eval_results/exp3

echo "=== Evaluating Experiment 4: YOLOv8 Deeper Neck ==="
python DeepDataMiningLearning/detection/mytrain_yolo.py \
    --model yolov8 \
    --scale m \
    --data-path $DATA_PATH \
    --annotationfile $ANNOTATION \
    --dataset waymococo \
    --resume ./outputs/exp4_yolov8_deeper_neck/checkpoint.pth \
    --test-only \
    --output-dir ./eval_results/exp4

echo "=== All evaluations complete! ==="
```

Make executable:
```bash
chmod +x evaluate_models.sh
./evaluate_models.sh
```

---

## Results Comparison Template

### Create results table

**File**: Create `results_comparison.md`:

```markdown
# Model Architecture Modification Results

## Dataset
- **Name**: Waymo Open Dataset (COCO format subset)
- **Images**: 4000 (step=5)
- **Classes**: Vehicle (1), Pedestrian (2), Cyclist (3), Sign (4)
- **Split**: 80% train, 20% val

## Experiments

### Baseline
- **Model**: Faster R-CNN ResNet50-FPN
- **Config**: Default torchvision, 256 FPN channels
- **Training**: 50 epochs, lr=0.001, batch=8

### Experiment 1: Deeper Backbone
- **Modification**: ResNet152 instead of ResNet50
- **Reasoning**: More layers for better feature extraction

| Metric | Baseline | Exp 1 | Change |
|--------|----------|-------|--------|
| mAP@0.5:0.95 | 0.XXX | 0.XXX | +X.X% |
| mAP@0.5 | 0.XXX | 0.XXX | +X.X% |
| AP (Vehicle) | 0.XXX | 0.XXX | +X.X% |
| AP (Pedestrian) | 0.XXX | 0.XXX | +X.X% |
| AP (Cyclist) | 0.XXX | 0.XXX | +X.X% |
| Parameters | 41M | 60M | +46% |
| Inference (ms) | 67 | 98 | +46% |

### Experiment 2: Increased FPN Capacity
- **Modification**: 512 FPN output channels (vs 256)
- **Reasoning**: Richer feature representation

| Metric | Baseline | Exp 2 | Change |
|--------|----------|-------|--------|
| mAP@0.5:0.95 | 0.XXX | 0.XXX | +X.X% |
| ... | ... | ... | ... |

## Best Configuration
Based on mAP@0.5:0.95 and inference speed trade-off:
- **Winner**: [Model name]
- **Justification**: [Why this is the best]

## Conclusion
[Summary of findings]
```

---

## Quick Start Commands

### 1. Train Baseline
```bash
python DeepDataMiningLearning/detection/mytrain.py \
    --model customrcnn_resnet50 \
    --data-path /path/to/waymo/ \
    --annotationfile /path/to/waymo/annotations.json \
    --dataset waymococo \
    --batch-size 8 \
    --epochs 50 \
    --output-dir ./outputs/baseline
```

### 2. Train Best Modification (ResNet152)
```bash
python DeepDataMiningLearning/detection/mytrain.py \
    --model customrcnn_resnet152 \
    --data-path /path/to/waymo/ \
    --annotationfile /path/to/waymo/annotations.json \
    --dataset waymococo \
    --batch-size 4 \
    --epochs 50 \
    --trainable 3 \
    --output-dir ./outputs/resnet152
```

### 3. Evaluate Both
```bash
# Baseline
python DeepDataMiningLearning/detection/mytrain.py \
    --model customrcnn_resnet50 \
    --data-path /path/to/waymo/ \
    --annotationfile /path/to/waymo/annotations.json \
    --dataset waymococo \
    --resume ./outputs/baseline/checkpoint.pth \
    --test-only

# Modified
python DeepDataMiningLearning/detection/mytrain.py \
    --model customrcnn_resnet152 \
    --data-path /path/to/waymo/ \
    --annotationfile /path/to/waymo/annotations.json \
    --dataset waymococo \
    --resume ./outputs/resnet152/checkpoint.pth \
    --test-only
```

---

## Tips for Success

1. **Start Simple**: Begin with Example 1 (ResNet152) - easy to implement, good results
2. **Monitor Training**: Use tensorboard or check loss curves
3. **Save Checkpoints**: Always save best models based on validation mAP
4. **Compare Fairly**: Use same dataset split, training epochs, and hyperparameters
5. **Document Everything**: Save training logs and evaluation results

---

Good luck with your experiments! 🚀
