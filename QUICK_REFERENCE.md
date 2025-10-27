# Architecture Modification Quick Reference

## 📚 Key Files for Modification

### Critical Files to Understand

1. **`detection/models.py`** (254 lines)
   - **Purpose**: Central model creation interface
   - **Key Function**: `create_detectionmodel()` - Creates any detection model
   - **Supports**: Faster R-CNN, YOLO, custom models
   - **Line 225-250**: CustomRCNN creation with configurable backbone

2. **`detection/backbone.py`** (174 lines)
   - **Purpose**: Backbone networks with FPN
   - **Class**: `MyBackboneWithFPN` - ResNet + FPN architecture
   - **Key Parameters**:
     - `model_name`: 'resnet50', 'resnet101', 'resnet152', 'swin_s'
     - `trainable_layers`: 0-5 (number of layers to fine-tune)
     - `out_channels`: 256 (default), can increase to 512+
   - **Line 30-90**: Main backbone+FPN construction

3. **`detection/modeling_rpnfasterrcnn.py`** (1310 lines)
   - **Purpose**: Complete Faster R-CNN implementation
   - **Class**: `CustomRCNN` (line 1100+) - Full detection model
   - **Key Components**:
     - Line 30-35: `AnchorGenerator` - anchor sizes/ratios
     - Line 250-450: `RegionProposalNetwork` - RPN logic
     - Line 600-900: `RoIHeads` - detection head
   - **Easy modifications**: Anchor parameters

4. **`detection/modules/yolomodels.py`** (2357 lines)
   - **Purpose**: YOLO architecture (v5, v7, v8, v11)
   - **Class**: `YoloDetectionModel` - Main YOLO class
   - **Key Methods**:
     - Line 140-200: `__init__()` - Model initialization
     - Line 300-400: `parse_model()` - Build layers from YAML
     - Line 500-550: `get_backbone()`, `get_neck()`, `get_head()`

5. **`detection/modules/block.py`** (~800 lines)
   - **Purpose**: Building blocks for YOLO
   - **Key Classes**:
     - `Conv`: Standard convolution
     - `C2f`: CSP bottleneck (line 200+)
     - `SPPF`: Spatial pyramid pooling
     - `Bottleneck`: Residual block

6. **`detection/modules/head.py`** (~500 lines)
   - **Purpose**: Detection heads
   - **Class**: `Detect` (line 30-150) - YOLOv8 detection head
   - **Easy to modify**: Add attention, change layers

7. **`detection/mytrain.py`** (591 lines)
   - **Purpose**: Training script for Faster R-CNN
   - **Key Function**: `main()` - Complete training loop
   - **Arguments**: Line 227-300
   - **Evaluation**: Line 450-550

8. **`detection/dataset_waymococo.py`** (530 lines)
   - **Purpose**: Waymo dataset in COCO format
   - **Class**: `WaymoCOCODataset`
   - **Classes**: Vehicle (1), Pedestrian (2), Cyclist (3), Sign (4)

---

## 🎯 Easiest Modifications (Ranked by Difficulty)

### ⭐ Level 1: Change Model Scale (No Code Modification)

**Just change command line argument:**
```bash
# Use bigger backbone
--model customrcnn_resnet152  # instead of resnet50

# Use bigger YOLO
--scale l  # instead of 'n' (nano) or 's' (small)

# More trainable layers
--trainable 3  # instead of 0 (frozen)
```

**Expected Impact**: +2-3% mAP, works immediately

---

### ⭐⭐ Level 2: Modify Single Parameter

**File**: `detection/modeling_rpnfasterrcnn.py`

**Change anchor ratios** (Line 35):
```python
aspect_ratios=((0.5, 0.75, 1.0, 1.5, 2.0),)  # Add 0.75, 1.5
```

**Expected Impact**: +1-2% mAP on small objects

---

### ⭐⭐⭐ Level 3: Increase Network Capacity

**File**: `detection/models.py` (Line 225)

**Change FPN channels**:
```python
out_channels=512,  # Change from 256
```

**File**: `detection/modules/yolov8.yaml`

**Increase C2f blocks**:
```yaml
- [-1, 6, C2f, [512]]  # Change 3 to 6
```

**Expected Impact**: +1-2% mAP

---

### ⭐⭐⭐⭐ Level 4: Add New Modules

**File**: Create `detection/modules/attention.py`

**Add CBAM attention**, then integrate in `detection/modules/head.py`

**Expected Impact**: +2-3% mAP

---

### ⭐⭐⭐⭐⭐ Level 5: Custom Architecture

Design completely new backbone or detection head from scratch.

---

## 🔥 Recommended Modifications for Your Project

### Option A: Faster R-CNN Based (Easier)

**Modification 1**: Deeper Backbone
```bash
--model customrcnn_resnet152
```

**Modification 2**: More Anchors
Edit `detection/modeling_rpnfasterrcnn.py` line 35:
```python
aspect_ratios=((0.5, 0.75, 1.0, 1.5, 2.0),)
```

**Why Good**:
- Easy to implement
- Clear improvement
- Well documented in torchvision

---

### Option B: YOLO Based (Faster Inference)

**Modification 1**: Increase Scale
```bash
--model yolov8 --scale m  # or 'l'
```

**Modification 2**: Deeper Neck
Edit `detection/modules/yolov8.yaml`:
```yaml
- [-1, 6, C2f, [512]]  # Change from 3 to 6
```

**Why Good**:
- YOLO trains faster
- Better real-time performance
- Modern architecture

---

## 📊 What to Report in Your Document

### 1. Architecture Description

**Template**:
```
Model: Faster R-CNN with ResNet152-FPN
Modifications:
1. Changed backbone from ResNet50 to ResNet152 (deeper network)
2. Increased trainable layers from 0 to 3 (fine-tuning more layers)
3. Added aspect ratios 0.75 and 1.5 to RPN anchors

Reasoning:
- Deeper backbone extracts more complex features
- Fine-tuning more layers adapts better to Waymo dataset
- More anchor aspect ratios detect pedestrians better (tall objects)
```

### 2. Training Setup

**Template**:
```
Dataset: Waymo Open Dataset (COCO format)
- Images: 4000 (step=5 subsample)
- Train/Val split: 80/20 (3200/800 images)
- Classes: 4 (Vehicle, Pedestrian, Cyclist, Sign)

Training Configuration:
- Epochs: 50
- Batch size: 4 (due to GPU memory)
- Learning rate: 0.001
- Optimizer: SGD with momentum=0.9
- Image size: 800x800
- Data augmentation: RandomHorizontalFlip, ColorJitter
```

### 3. Results Table

**Template**:
```
| Model | mAP@0.5:0.95 | mAP@0.5 | AP_Vehicle | AP_Ped | AP_Cyclist | Params | Speed |
|-------|--------------|---------|------------|--------|------------|--------|-------|
| Baseline (ResNet50) | 0.325 | 0.512 | 0.448 | 0.256 | 0.271 | 41M | 15 fps |
| Modified (ResNet152) | 0.351 | 0.538 | 0.475 | 0.283 | 0.295 | 60M | 10 fps |
| Improvement | +8.0% | +5.1% | +6.0% | +10.5% | +8.9% | +46% | -33% |
```

### 4. Analysis

**Template**:
```
Results Analysis:

The modified ResNet152 model achieves 8.0% improvement in mAP@0.5:0.95 
compared to the baseline ResNet50 model. Key observations:

1. Best improvement on Pedestrian class (+10.5%)
   - Reason: Deeper features help detect small objects
   
2. Modest improvement on Vehicle class (+6.0%)
   - Reason: Vehicles already well-detected in baseline
   
3. Trade-offs:
   - 46% more parameters (60M vs 41M)
   - 33% slower inference (10fps vs 15fps)
   - Acceptable for autonomous vehicles prioritizing accuracy

Conclusion: The deeper backbone significantly improves detection,
especially for challenging small objects, at a reasonable 
computational cost.
```

---

## 🚀 Step-by-Step Workflow

### Week 1: Setup and Baseline

```bash
# Day 1-2: Setup environment and data
cd DeepDataMiningLearning
python -m pip install -r requirements.txt

# Day 3-4: Train baseline
python detection/mytrain.py \
    --model customrcnn_resnet50 \
    --data-path /data/waymo/ \
    --dataset waymococo \
    --epochs 50 \
    --output-dir ./outputs/baseline

# Day 5: Evaluate baseline
python detection/mytrain.py \
    --model customrcnn_resnet50 \
    --data-path /data/waymo/ \
    --dataset waymococo \
    --resume ./outputs/baseline/checkpoint.pth \
    --test-only
```

### Week 2: Modifications

```bash
# Day 1-3: Modification 1 - ResNet152
python detection/mytrain.py \
    --model customrcnn_resnet152 \
    --data-path /data/waymo/ \
    --dataset waymococo \
    --epochs 50 \
    --output-dir ./outputs/mod1_resnet152

# Day 4-5: Modification 2 - Edit code for anchors, retrain

# Weekend: Evaluate and compare results
```

### Week 3: Analysis and Report

- Compare mAP metrics
- Create visualization plots
- Write report with architecture diagrams
- Prepare presentation

---

## 🔍 Understanding mAP Output

When you run evaluation, you'll see:

```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.325
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.512
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.341
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.186
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.401
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.558
```

**What to report**:
- **Primary metric**: AP @ IoU=0.50:0.95 (0.325 in this example)
- **Comparison**: Compare baseline vs modified
- **Per-class AP**: Check which classes improved
- **Size-specific AP**: small/medium/large objects

---

## 🎨 Visualization Tips

### Generate Detection Images

Add to your evaluation script:

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_detections(model, dataset, num_images=5):
    model.eval()
    for i in range(num_images):
        img, target = dataset[i]
        with torch.no_grad():
            prediction = model([img])
        
        # Plot image with boxes
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(img.permute(1, 2, 0))
        
        # Draw predictions
        for box, label, score in zip(prediction[0]['boxes'], 
                                      prediction[0]['labels'], 
                                      prediction[0]['scores']):
            if score > 0.5:  # Confidence threshold
                rect = patches.Rectangle((box[0], box[1]), 
                                        box[2]-box[0], box[3]-box[1],
                                        linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        
        plt.savefig(f'detection_result_{i}.jpg')
```

---

## 📝 Common Mistakes to Avoid

1. **Wrong num_classes**: Must be 5 for Waymo (background + 4 objects)
2. **Batch size too large**: Start with 4-8, increase if no OOM
3. **Not saving checkpoints**: Always use `--output-dir`
4. **Comparing different epochs**: Train all models for same epochs
5. **Ignoring validation**: Monitor validation mAP during training

---

## 🆘 Troubleshooting

### Issue: Out of Memory (OOM)
**Solution**:
```bash
--batch-size 2  # Reduce batch size
--trainable 0   # Freeze more layers
```

### Issue: NaN Loss
**Solution**:
```bash
--lr 0.0001     # Lower learning rate
--clip-grad-norm 1.0  # Gradient clipping
```

### Issue: Class Mismatch Error
**Solution**: Check `num_classes` matches dataset (5 for Waymo with background)

### Issue: No Improvement
**Solution**: 
- Train longer (100 epochs)
- Check data augmentation
- Verify dataset quality
- Try different learning rate

---

## ✅ Final Checklist

Before submitting:

- [ ] Trained baseline model
- [ ] Trained at least 2 modified models
- [ ] Evaluated all models on same test set
- [ ] Recorded mAP@0.5:0.95 for all models
- [ ] Documented architecture changes
- [ ] Explained reasoning for modifications
- [ ] Compared results in table format
- [ ] Generated visualization plots
- [ ] Checked model parameters and speed
- [ ] Written analysis of results
- [ ] Included training commands
- [ ] Saved model checkpoints

---

## 🎓 Additional Reading

- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144)
- [COCO Detection Metrics](https://cocodataset.org/#detection-eval)
- [Waymo Open Dataset](https://waymo.com/open/)

---

## 💡 Pro Tips

1. **Start with easiest modification first** (ResNet152) to ensure pipeline works
2. **Save checkpoints every 10 epochs** in case training crashes
3. **Use tensorboard** to monitor training: `tensorboard --logdir ./outputs`
4. **Compare with published baselines** from Waymo leaderboard
5. **Document everything** as you go, don't wait until end

---

**Good luck! You have everything you need to succeed! 🚀**

Questions? Check:
- `ARCHITECTURE_MODIFICATION_GUIDE.md` - Comprehensive guide
- `EXAMPLE_MODIFICATIONS.md` - Concrete code examples
- This file - Quick reference and tips
