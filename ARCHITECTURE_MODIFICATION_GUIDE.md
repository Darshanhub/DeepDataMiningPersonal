# Object Detection Model Architecture Modification Guide

## Overview
This guide explains how to modify model architectures in the DeepDataMiningLearning repository for object detection tasks on Waymo or NuScenes datasets.

---

## 📁 Repository Structure for Detection

```
DeepDataMiningLearning/
├── detection/
│   ├── models.py                    # Main model creation interface
│   ├── backbone.py                  # Backbone architectures (ResNet, etc.)
│   ├── modeling_rpnfasterrcnn.py   # Custom Faster R-CNN implementation
│   ├── modeling_yolo.py             # YOLO model wrapper
│   ├── mytrain.py                   # Training script for Faster R-CNN
│   ├── mytrain_yolo.py              # Training script for YOLO
│   ├── dataset_waymococo.py         # Waymo dataset in COCO format
│   ├── dataset_nuscenes.py          # NuScenes dataset loader
│   └── modules/
│       ├── yolomodels.py            # YOLO architecture implementation
│       ├── head.py                  # Detection heads
│       ├── block.py                 # Building blocks (Conv, Bottleneck, etc.)
│       └── anchor.py                # Anchor generation
```

---

## 🎯 Available Model Architectures

### 1. **Faster R-CNN Based Models**
- **Model Name**: `fasterrcnn_resnet50_fpn_v2`, `customrcnn_resnet50`, `customrcnn_resnet152`
- **Location**: `detection/modeling_rpnfasterrcnn.py`, `detection/models.py`
- **Components**:
  - Backbone: ResNet (50, 101, 152) with FPN
  - RPN (Region Proposal Network)
  - ROI Heads with box predictor

### 2. **YOLO Models**
- **Model Name**: `yolov8`, `yolov7`, `yolov5`, `yolov11`
- **Location**: `detection/modules/yolomodels.py`
- **Components**:
  - Backbone: CSPDarknet / CSP-based
  - Neck: PANet/FPN
  - Detection Head: Decoupled head

### 3. **Custom Torchvision YOLO**
- **Model Name**: `torchvisionyolo`
- **Location**: `detection/modeling_yolo.py`
- **Scale Options**: 'n', 's', 'm', 'l', 'x'

---

## 🔧 How to Modify Model Architecture

### Option 1: Modify Faster R-CNN Backbone

**File**: `detection/backbone.py`

#### Current Backbone Options:
```python
# Available backbones
model_names = ['resnet50', 'resnet101', 'resnet152', 'swin_s', 'swin_t']
```

#### Modification Steps:

**1. Change Backbone Network**

Edit `MyBackboneWithFPN` class in `detection/backbone.py`:

```python
class MyBackboneWithFPN(nn.Module):
    def __init__(
        self,
        model_name: str,  # Change this: 'resnet50', 'resnet152', 'swin_s'
        trainable_layers: int,  # 0-5: layers to fine-tune
        out_channels: int = 256,
        ...
    ):
```

**Key modification points:**
- Line ~40-45: Backbone initialization
- Line ~50-55: Trainable layers configuration
- Line ~60-75: FPN layer configuration

**2. Modify FPN Channels**

```python
# In MyBackboneWithFPN.__init__()
out_channels = 512  # Change from 256 to increase capacity
```

**3. Change Feature Pyramid Levels**

```python
# Line ~60
returned_layers = [1, 2, 3, 4]  # Can change to [2, 3, 4] for fewer scales
```

**4. Modify RPN Configuration**

Edit `detection/modeling_rpnfasterrcnn.py`:

```python
# Line ~30-35: Change anchor sizes
def __init__(
    self,
    sizes=((32, 64, 128, 256, 512),),  # Modify anchor sizes
    aspect_ratios=((0.5, 1.0, 2.0),),  # Modify aspect ratios
):
```

---

### Option 2: Modify YOLO Architecture

**File**: `detection/modules/yolomodels.py`

#### Architecture Components:

**1. Backbone Modification**

Edit YAML configuration file (e.g., `detection/modules/yolov8.yaml`):

```yaml
# YOLOv8 backbone
backbone:
  # Conv layers
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2  [out_channels, kernel, stride]
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C2f, [128, True]]  # 2 [channels, bottleneck]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C2f, [256, True]]  # 4
  # ... add/modify layers
```

**2. Modify C2f Blocks** (Key bottleneck structure)

Edit `detection/modules/block.py`:

```python
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels - MODIFY THIS RATIO
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
```

**3. Modify Detection Head**

Edit `detection/modules/head.py`:

```python
class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers (3)
        self.reg_max = 16  # DFL channels - INCREASE for better localization
        # ... modify layers
```

**4. Change Neck Architecture**

Edit the neck section in YAML:

```yaml
# YOLOv8 neck
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # Modify number of blocks (3->6 for deeper)
```

---

### Option 3: Add Custom Modules

**Create a new detection head:**

1. **Add new head class** in `detection/modules/head.py`:

```python
class CustomDetectHead(nn.Module):
    """Custom detection head with attention."""
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        
        # Add attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        
        # Detection layers
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, 256, 3), Conv(256, 4 * self.reg_max, 1))
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, 256, 3), Conv(256, self.nc, 1))
            for x in ch
        )
    
    def forward(self, x):
        # Apply attention
        for i in range(self.nl):
            x[i] = self.attention(x[i])
        # ... detection logic
```

2. **Register in model creation**:

Edit `detection/modules/yolomodels.py`:

```python
# In parse_model function, add:
elif m in (CustomDetectHead,):
    args = [nc, [ch[x] for x in f]]
```

---

## 🏋️ Training Models on Waymo/NuScenes

### Dataset Preparation

#### Waymo Dataset (COCO Format)
```bash
# Dataset structure
/path/to/waymo/
├── images/
│   ├── segment_xxx_xxx_frame_000.jpg
│   └── ...
└── annotations.json  # COCO format
```

**Dataset loader**: `detection/dataset_waymococo.py`

Classes:
- 1: Vehicle
- 2: Pedestrian  
- 3: Cyclist
- 4: Sign

#### NuScenes Dataset
```bash
# Dataset structure
/path/to/nuscenes/
├── samples/
│   ├── CAM_FRONT/
│   └── ...
└── v1.0-trainval/
```

**Dataset loader**: `detection/dataset_nuscenes.py`

### Training Commands

#### Train Faster R-CNN on Waymo

```bash
python DeepDataMiningLearning/detection/mytrain.py \
    --model customrcnn_resnet152 \
    --data-path /path/to/waymo_subset_coco/ \
    --annotationfile /path/to/waymo_subset_coco/annotations.json \
    --dataset waymococo \
    --batch-size 8 \
    --epochs 50 \
    --lr 0.001 \
    --trainable 3 \
    --output-dir ./outputs/waymo_resnet152
```

**Key arguments:**
- `--model`: Choose architecture (`customrcnn_resnet50`, `customrcnn_resnet152`, `fasterrcnn_resnet50_fpn_v2`)
- `--trainable`: Number of trainable backbone layers (0-5)
- `--nocustomize`: Use pretrained head without modification

#### Train YOLO on Waymo

```bash
python DeepDataMiningLearning/detection/mytrain_yolo.py \
    --model yolov8 \
    --scale n \
    --data-path /path/to/waymo_subset_coco/ \
    --annotationfile /path/to/waymo_subset_coco/annotations.json \
    --dataset waymococo \
    --batch-size 16 \
    --epochs 100 \
    --lr 0.01 \
    --output-dir ./outputs/waymo_yolov8n
```

**Scale options**: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra-large)

#### Train on NuScenes

```bash
python DeepDataMiningLearning/detection/mytrain.py \
    --model customrcnn_resnet152 \
    --data-path /path/to/nuscenes/ \
    --dataset nuscenes \
    --batch-size 8 \
    --epochs 50 \
    --output-dir ./outputs/nuscenes_resnet152
```

---

## 📊 Evaluation and mAP Calculation

### Evaluate Trained Model

```bash
python DeepDataMiningLearning/detection/mytrain.py \
    --model customrcnn_resnet152 \
    --data-path /path/to/waymo_subset_coco/ \
    --annotationfile /path/to/waymo_subset_coco/annotations.json \
    --dataset waymococo \
    --resume /path/to/checkpoint.pth \
    --test-only \
    --output-dir ./evaluation_results
```

### mAP Evaluation Output

The evaluation will output:
```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.xxx
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.xxx
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.xxx
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.xxx
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.xxx
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.xxx
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.xxx
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.xxx
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.xxx
```

**Key metrics:**
- **AP @ IoU=0.50:0.95**: Primary COCO metric (average over IoU thresholds)
- **AP @ IoU=0.50**: Pascal VOC metric
- **AP @ IoU=0.75**: Strict localization metric

---

## 🚀 Suggested Architecture Modifications

### 1. **Deeper Backbone**
Change from ResNet50 to ResNet152:
```python
# In mytrain.py command
--model customrcnn_resnet152
```

**Expected impact**: +2-3% mAP, slower inference

### 2. **Larger FPN Channels**
Edit `detection/backbone.py`:
```python
out_channels = 512  # Instead of 256
```

**Expected impact**: +1-2% mAP, +30% parameters

### 3. **More RPN Anchors**
Edit `detection/modeling_rpnfasterrcnn.py`:
```python
sizes=((16, 32, 64, 128, 256, 512),)  # Add smaller anchor
aspect_ratios=((0.5, 0.75, 1.0, 1.5, 2.0),)  # More ratios
```

**Expected impact**: +1-2% mAP on small objects

### 4. **Attention in YOLO Neck**
Add attention module in `detection/modules/block.py`:
```python
class C2fWithAttention(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.attn = nn.MultiheadAttention(c2, num_heads=8)
```

**Expected impact**: +1-3% mAP, +10% compute

### 5. **Deformable Convolutions**
Replace standard Conv in backbone:
```python
from torchvision.ops import DeformConv2d

class DeformableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2*kernel_size*kernel_size, kernel_size)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size)
```

**Expected impact**: +2-4% mAP on deformed objects

---

## 📋 Model Comparison Template

### Experiment Tracking

| Model | Backbone | Modifications | Dataset | mAP@0.5 | mAP@0.5:0.95 | Params | Speed (fps) |
|-------|----------|---------------|---------|---------|--------------|--------|-------------|
| Baseline | ResNet50-FPN | None | Waymo 4k | - | - | 41M | 15 |
| Modified 1 | ResNet152-FPN | Deeper backbone | Waymo 4k | - | - | 60M | 10 |
| Modified 2 | ResNet50-FPN | 512 FPN channels | Waymo 4k | - | - | 55M | 12 |
| Modified 3 | YOLOv8m | Default | Waymo 4k | - | - | 25M | 45 |
| Modified 4 | YOLOv8m | +Attention neck | Waymo 4k | - | - | 28M | 38 |

---

## 🐛 Common Issues and Solutions

### Issue 1: Class Mismatch
**Error**: `RuntimeError: shape mismatch`

**Solution**: Ensure `num_classes` matches dataset:
```python
# For Waymo: 5 classes (background + 4 objects)
model = create_detectionmodel('customrcnn_resnet50', num_classes=5)
```

### Issue 2: Out of Memory
**Solution**: Reduce batch size or image size:
```bash
--batch-size 4 --img-size 640
```

### Issue 3: NaN Loss
**Solution**: Lower learning rate or use gradient clipping:
```bash
--lr 0.0001 --clip-grad-norm 1.0
```

---

## 📚 Key Files Reference

### Model Creation
- `detection/models.py`: `create_detectionmodel()` - Main interface
- `detection/backbone.py`: `MyBackboneWithFPN` - Custom backbone with FPN
- `detection/modeling_rpnfasterrcnn.py`: `CustomRCNN` - Complete Faster R-CNN

### Training
- `detection/mytrain.py`: Faster R-CNN training loop
- `detection/mytrain_yolo.py`: YOLO training loop
- `detection/trainutils.py`: Training utilities

### Evaluation  
- `detection/myevaluator.py`: `modelevaluate()` - COCO evaluation
- `detection/cocoevaluator.py`: Detailed COCO metrics

### Dataset
- `detection/dataset.py`: `get_dataset()` - Dataset factory
- `detection/dataset_waymococo.py`: `WaymoCOCODataset` 
- `detection/dataset_nuscenes.py`: NuScenes loader

---

## 🎓 Example Modification Workflow

### Step 1: Choose Base Model
```bash
# Start with baseline
python detection/mytrain.py --model customrcnn_resnet50 --data-path /data/waymo
```

### Step 2: Modify Architecture
Edit `detection/backbone.py`:
```python
# Change to ResNet152
backbone = resnet.__dict__['resnet152'](weights=weights)
```

### Step 3: Train Modified Model
```bash
python detection/mytrain.py \
    --model customrcnn_resnet152 \
    --data-path /data/waymo \
    --epochs 50 \
    --output-dir ./outputs/modified
```

### Step 4: Evaluate and Compare
```bash
python detection/mytrain.py \
    --model customrcnn_resnet152 \
    --data-path /data/waymo \
    --resume ./outputs/modified/checkpoint.pth \
    --test-only
```

### Step 5: Document Results
Record mAP, inference speed, and model size for comparison.

---

## 📖 Additional Resources

- **Torchvision Detection Reference**: https://pytorch.org/vision/stable/models.html#object-detection
- **YOLO Architecture**: https://docs.ultralytics.com/models/
- **COCO Evaluation Metrics**: https://cocodataset.org/#detection-eval
- **Waymo Dataset**: https://waymo.com/open/
- **NuScenes Dataset**: https://www.nuscenes.org/

---

## ✅ Checklist for Architecture Modifications

- [ ] Choose base model (Faster R-CNN vs YOLO)
- [ ] Identify modification target (backbone, neck, head)
- [ ] Modify architecture files
- [ ] Verify model creation (no errors)
- [ ] Train baseline model
- [ ] Train modified model  
- [ ] Evaluate both models (mAP@0.5, mAP@0.5:0.95)
- [ ] Measure inference speed
- [ ] Count parameters
- [ ] Document changes and results
- [ ] Compare performance improvements

---

**Good luck with your model modifications! 🚀**
