# Deepfake Detection System - Architectural Design

## Table of Contents
1. [System Overview](#system-overview)
2. [Data Flow Architecture](#data-flow-architecture)
3. [Preprocessing Pipeline](#preprocessing-pipeline)
4. [Dataset Architecture](#dataset-architecture)
5. [Model Architecture](#model-architecture)
6. [Training Pipeline](#training-pipeline)
7. [Inference Systems](#inference-systems)
8. [Evaluation Framework](#evaluation-framework)

---

## System Overview

### Purpose
A temporal deepfake detection system that analyzes video sequences to identify manipulated (fake) videos from real ones using deep learning.

### Key Components
```
┌─────────────────────────────────────────────────────────────┐
│                    Raw Video Dataset                        │
│              (CelebDF: Real + Synthesis)                    │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Preprocessing Pipeline                         │
│    (Face Detection → Sampling → Windowing)                  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│           Processed Dataset (.pt files)                     │
│    (Windowed Face Sequences + Metadata)                     │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│               PyTorch Dataset Loader                        │
│         (DeepfakeWindowDataset)                             │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│            ResNeXt50 + GRU Model                            │
│    (Spatial Feature Extraction + Temporal Modeling)         │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Training & Validation                          │
│    (BCEWithLogitsLoss + Adam + Mixed Precision)             │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│         Trained Model Checkpoints                           │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│    Inference & Evaluation Systems                           │
│  (Batch Inference, Single Video, Metrics)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow Architecture

### Directory Structure
```
DeepFake/
│
├── CelebDf_new/                    # Raw dataset (input)
│   ├── Celeb-real/
│   ├── Celeb-synthesis/
│   └── YouTube-real/
│
├── celebdf_split/                  # Train/Val/Test split
│   ├── train/
│   │   ├── Celeb-real/
│   │   ├── Celeb-synthesis/
│   │   └── YouTube-real/
│   ├── val/
│   └── test/
│
├── celebdf_processed_data/         # Preprocessed tensors
│   ├── train/
│   │   ├── Celeb-real/*.pt
│   │   ├── Celeb-synthesis/*.pt
│   │   └── YouTube-real/*.pt
│   ├── val/
│   └── test/
│
├── checkpoints/                    # Model checkpoints
│   ├── model_epoch_1.pth
│   ├── model_epoch_2.pth
│   └── best_model.pth
│
└── [evaluation outputs]/           # Metrics and predictions
    ├── test_metrics_summary.json
    ├── test_detailed_predictions.json
    └── evaluation_plots.png
```

### Data Flow Stages

1. **Raw Video → Split** (Manual/External)
   - Input: MP4 videos
   - Output: Train/Val/Test directories

2. **Split → Preprocessed** (preprocess.py + train_process.py)
   - Input: MP4 videos from celebdf_split/
   - Processing: Face detection, sampling, windowing
   - Output: .pt files in celebdf_processed_data/

3. **Preprocessed → Training** (dataset.py + train_model.py)
   - Input: .pt tensor files
   - Processing: DataLoader → Model → Optimization
   - Output: Model checkpoints

4. **Model → Inference** (inference.py, batch_inference.py)
   - Input: Trained model + videos/tensors
   - Processing: Prediction with temporal analysis
   - Output: Predictions, confidence scores, segments

5. **Model → Evaluation** (evaluate.py, evaluate_new.py)
   - Input: Trained model + test data
   - Processing: Metrics calculation, visualization
   - Output: JSON reports, plots

---

## Preprocessing Pipeline

### Architecture (`preprocess.py`)

#### Component: VideoPreprocessor Class

**Purpose**: Convert raw videos into normalized face sequence tensors suitable for temporal deepfake detection.

**Key Parameters**:
```python
sequence_length = 30    # Frames per window (10 seconds at 3 FPS)
target_fps = 3          # Sampling rate (reduce redundancy)
image_size = 224        # Face crop size (ResNet standard)
```

### Processing Stages

#### Stage 1: Face Detection
```
Technology: MTCNN (Multi-task Cascaded Convolutional Networks)
Configuration:
  - keep_all=False         # Focus on primary face
  - select_largest=True    # Handle multiple faces
  - device='cuda'          # GPU acceleration
  - image_size=224         # Output resolution
  - margin=0               # Tight crop
```

**Workflow**:
1. Read video frame (BGR format from OpenCV)
2. Convert BGR → RGB → PIL Image
3. Pass through MTCNN detector
4. Return aligned face tensor (3, 224, 224)
5. Handle failures gracefully (skip frames without clear faces)

#### Stage 2: Frame Sampling
```
Strategy: Fixed-rate sampling at 3 FPS
Formula: frame_interval = original_fps / target_fps

Example:
  30 FPS video → Sample every 10th frame
  24 FPS video → Sample every 8th frame
  
Timestamp Calculation:
  timestamp = frame_index / original_fps
```

**Benefits**:
- Reduces temporal redundancy (adjacent frames are nearly identical)
- Decreases processing time by 10x (30 FPS → 3 FPS)
- Maintains temporal dynamics for manipulation detection
- Standardizes across different source frame rates

#### Stage 3: Temporal Windowing
```
Configuration:
  Window Size: 30 frames (10 seconds)
  Overlap: 6 frames (2 seconds)
  Stride: 24 frames (8 seconds)

Sliding Window Calculation:
  for i in range(0, num_frames - 30 + 1, stride=24):
      window = faces[i : i + 30]
      
Leftover Handling:
  If remaining frames < 30:
      Use last 30 frames of video (ensures coverage)
```

**Example**:
```
Video: 15 seconds at 3 FPS = 45 frames

Window 1: Frames 0-29   (0s - 9.67s)
Window 2: Frames 24-53  (8s - 17.67s)  ← 6-frame overlap with Window 1
Last:     Frames 15-44  (5s - 14.67s)  ← Ensures end coverage
```

#### Stage 4: Normalization & Storage
```
Tensor Shape: (num_windows, 30, 3, 224, 224)
  - num_windows: Variable per video
  - 30: Sequence length
  - 3: RGB channels
  - 224x224: Spatial dimensions

Storage Format (.pt file):
{
    'windows': Tensor,              # Face sequences
    'timestamps': [(start, end)],   # Time ranges
    'num_windows': int,             # Count
    'sequence_length': 30,
    'target_fps': 3,
    'image_size': 224
}
```

### Design Rationale

| Decision | Rationale |
|----------|-----------|
| 3 FPS sampling | Balances temporal resolution with efficiency; captures changes while avoiding redundancy |
| 30-frame windows | 10 seconds provides enough context for temporal inconsistency detection |
| 2-second overlap | Ensures smooth transitions and prevents boundary artifacts |
| MTCNN detector | State-of-the-art face alignment; handles pose variations |
| 224x224 resolution | Standard ImageNet size; compatible with pretrained models |
| .pt format | Native PyTorch serialization; fast loading with metadata |

---

## Dataset Architecture

### Component: DeepfakeWindowDataset (`dataset.py`)

**Purpose**: PyTorch Dataset interface for loading preprocessed video windows with labels.

### Class Design

```python
class DeepfakeWindowDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        # Loads all .pt files and creates index of windows
        
    def __len__(self):
        # Returns total number of windows across all videos
        
    def __getitem__(self, idx):
        # Returns (window_tensor, label) for specific index
```

### Label Mapping Strategy

```
Real Videos (Label = 0.0):
  - Celeb-real/        # Celebrity real videos
  - YouTube-real/      # YouTube real videos

Fake Videos (Label = 1.0):
  - Celeb-synthesis/   # Deepfake videos
```

### Indexing Architecture

**Problem**: Videos have variable numbers of windows. How to create flat index?

**Solution**: Pre-compute index mapping
```python
self.samples = []  # List of (file_path, window_index, label)

For each .pt file:
    num_windows = data['num_windows']
    For each window in file:
        if window has exactly 30 frames:  # Quality filter
            samples.append((file_path, window_idx, label))
```

**Example**:
```
video1.pt: 5 windows → Index 0-4
video2.pt: 3 windows → Index 5-7
video3.pt: 7 windows → Index 8-14

Total dataset size: 15 windows
```

### Quality Filtering

```python
# Only include windows with exactly 30 frames
if data['windows'][i].shape[0] == 30:
    self.samples.append((f, i, label))
```

**Rationale**: Prevents tensor size mismatches during batching; ensures consistent temporal context.

### Data Loading Optimization

```python
def create_dataloaders(data_dir, batch_size=8, num_workers=2):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,              # Randomize training
        num_workers=num_workers,   # Parallel loading
        pin_memory=True           # Faster GPU transfer
    )
```

**Configuration Choices**:
- `batch_size=8`: Balance between GPU memory and throughput
- `num_workers=4-6`: Parallel data loading (adjust based on CPU cores)
- `pin_memory=True`: Enables async GPU transfer
- `persistent_workers=True`: Reuses worker processes (faster epochs)

---

## Model Architecture

### Component: ResNeXt50 + GRU Hybrid Model

**Purpose**: Combine spatial feature extraction (CNN) with temporal modeling (RNN) for video-level deepfake detection.

### Architecture Diagram

```
Input: (batch_size, 30, 3, 224, 224)
       [Batch of 30-frame sequences]
         │
         ▼
┌─────────────────────────────────┐
│     ResNeXt50 Feature Extractor │
│   (Pretrained on ImageNet)      │
│                                 │
│   Per-frame processing:         │
│   (30, 3, 224, 224) → (30, 2048)│
└─────────────┬───────────────────┘
              │
              ▼ [2048-D features per frame]
┌─────────────────────────────────┐
│      2-Layer Bidirectional GRU  │
│                                 │
│   Hidden Size: 512              │
│   Layers: 2                     │
│   Dropout: 0.3                  │
│                                 │
│   Captures temporal patterns    │
└─────────────┬───────────────────┘
              │
              ▼ [1024-D temporal features]
┌─────────────────────────────────┐
│      Fully Connected Layers     │
│                                 │
│   FC1: 1024 → 512 (+ ReLU + BN) │
│   Dropout: 0.5                  │
│   FC2: 512 → 1 (logit)          │
└─────────────┬───────────────────┘
              │
              ▼
         Single Logit
    (BCEWithLogitsLoss)
         │
         ▼ [During inference]
      Sigmoid(logit)
         │
         ▼
    Probability [0, 1]
```

### Component Breakdown

#### 1. ResNeXt50 CNN Backbone
```python
Technology: torchvision.models.resnext50_32x4d
Pretrained: ImageNet (1000 classes)
Modification: Remove final FC layer
Output: 2048-dimensional feature vectors

Configuration Options:
  - freeze_cnn=True:  Freeze weights, use as feature extractor
  - freeze_cnn=False: Fine-tune on deepfake task
```

**Design Rationale**:
- ResNeXt > ResNet: Better feature representation via group convolutions
- Pretrained weights: Transfer learning from ImageNet
- 2048-D features: Rich spatial representations

#### 2. Bidirectional GRU (Gated Recurrent Unit)
```python
Configuration:
  input_size = 2048        # From CNN
  hidden_size = 512        # Per direction
  num_layers = 2           # Stacked layers
  bidirectional = True     # Forward + backward context
  dropout = 0.3            # Inter-layer regularization
  batch_first = True       # (batch, seq, features)

Output Size: 512 * 2 = 1024  # Concatenated bidirectional
```

**Why GRU over LSTM?**:
- Simpler architecture (fewer parameters)
- Faster training
- Comparable performance on sequential tasks
- Less prone to overfitting on smaller datasets

**Why Bidirectional?**:
- Captures both past and future context
- Important for detecting temporal inconsistencies
- Manipulation artifacts can precede/follow in time

#### 3. Classification Head
```python
Architecture:
  FC1: 1024 → 512
  BatchNorm1d(512)
  ReLU activation
  Dropout(0.5)
  FC2: 512 → 1

Output: Single logit (no sigmoid in model)
```

**Design Choices**:
- Batch Normalization: Stabilizes training, allows higher learning rates
- Dropout(0.5): Prevents overfitting in FC layers
- No sigmoid: Let BCEWithLogitsLoss handle it (more numerically stable)

### Forward Pass Flow

```python
def forward(self, x):
    # Input: (B, T, C, H, W) = (batch, 30, 3, 224, 224)
    
    # Step 1: Reshape for CNN
    B, T, C, H, W = x.shape
    x = x.view(B * T, C, H, W)  # (B*30, 3, 224, 224)
    
    # Step 2: Extract features
    features = self.cnn(x)  # (B*30, 2048)
    
    # Step 3: Reshape for GRU
    features = features.view(B, T, -1)  # (B, 30, 2048)
    
    # Step 4: Temporal modeling
    gru_out, _ = self.gru(features)  # (B, 30, 1024)
    
    # Step 5: Use last timestep
    final_features = gru_out[:, -1, :]  # (B, 1024)
    
    # Step 6: Classification
    x = self.fc1(final_features)  # (B, 512)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.dropout(x)
    logit = self.fc2(x)  # (B, 1)
    
    return logit  # Raw logit (not probability)
```

### Training Configuration

```python
Loss Function: BCEWithLogitsLoss
  - Combines sigmoid + BCE in numerically stable way
  - Supports mixed precision training (AMP)
  
Optimizer: Adam
  - Learning rate: 1e-4
  - Fused kernel (faster on GPU)
  
Scheduler: ReduceLROnPlateau
  - Monitors validation loss
  - Reduces LR by 0.5 when plateau detected
  - Patience: 2 epochs
  
Mixed Precision: Automatic Mixed Precision (AMP)
  - FP16 for faster training
  - GradScaler for numerical stability
```

---

## Training Pipeline

### Component: train_model.py

**Purpose**: End-to-end training loop with validation, checkpointing, and automatic evaluation.

### Training Architecture

```
┌─────────────────────────────────────────────────────────┐
│               Initialization Phase                      │
│  - Load datasets (train/val/test)                       │
│  - Create model (ResNeXt50+GRU)                         │
│  - Setup optimizer, scheduler, loss                     │
│  - Initialize AMP scaler                                │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                  Epoch Loop                             │
│  For epoch in range(num_epochs):                        │
│    1. Train Phase                                       │
│    2. Validation Phase                                  │
│    3. Checkpoint Saving                                 │
│    4. Learning Rate Scheduling                          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│            Automatic Evaluation                         │
│  - Load best model                                      │
│  - Evaluate on test set                                 │
│  - Generate metrics and plots                           │
└─────────────────────────────────────────────────────────┘
```

### Training Loop (train_epoch function)

```python
def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()  # Enable dropout, batch norm training mode
    
    for inputs, labels in dataloader:
        # Transfer to GPU (non-blocking for async)
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Zero gradients efficiently
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision forward pass
        with autocast('cuda'):
            outputs = model(inputs).squeeze()  # (B, 1) → (B,)
            loss = criterion(outputs, labels)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate accuracy
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        accuracy = (predicted == labels).float().mean()
```

**Key Optimizations**:
1. **Mixed Precision (AMP)**: 
   - Uses FP16 for computation (2x faster)
   - Maintains FP32 for critical operations
   - GradScaler prevents underflow

2. **Efficient Memory Management**:
   - `zero_grad(set_to_none=True)`: Deallocates gradient memory
   - `non_blocking=True`: Async GPU transfer
   - `pin_memory=True`: Faster CPU→GPU transfer

3. **Parallel Data Loading**:
   - `num_workers=4-6`: Prefetch batches in parallel
   - `persistent_workers=True`: Reuse worker processes

### Validation Loop

```python
def validate(model, dataloader, criterion, device):
    model.eval()  # Disable dropout, fix batch norm
    
    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            
            # Calculate metrics
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            # Accumulate accuracy, loss
```

### Checkpointing Strategy

```python
# Save every epoch
checkpoint = {
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_acc': val_acc
}
torch.save(checkpoint, f'checkpoints/model_epoch_{epoch+1}.pth')

# Save best model separately
if val_acc > best_val_acc:
    torch.save(model.state_dict(), 'checkpoints/best_model.pth')
```

### Training Hyperparameters

```python
Default Configuration:
  batch_size = 4           # Adjust based on GPU memory
  num_epochs = 10          # Typical for transfer learning
  learning_rate = 1e-4     # Conservative for fine-tuning
  freeze_cnn = False       # Fine-tune CNN (slower but better)
  use_amp = True           # Mixed precision (faster)
  num_workers = 4          # Parallel data loading

Speed Optimizations (Optional):
  freeze_cnn = True        # 2-3x faster training
  batch_size = 8           # Larger batches (if GPU allows)
  num_workers = 6          # More prefetch workers
```

---

## Inference Systems

### 1. Single Video Inference (`inference.py`)

**Purpose**: Predict on individual videos with temporal segmentation of manipulated regions.

#### Architecture
```python
def predict_video(video_path, model, preprocessor, device):
    # Stage 1: Preprocess video
    windows, timestamps = preprocessor.process_video(video_path)
    
    # Stage 2: Predict each window
    window_predictions = []
    for window in windows:
        logit = model(window)
        confidence = sigmoid(logit)
        window_predictions.append(confidence)
    
    # Stage 3: Temporal smoothing
    smoothed = sliding_window_smoothing(window_predictions)
    
    # Stage 4: Segment detection
    segments = identify_manipulated_segments(smoothed, timestamps)
    
    # Stage 5: Overall decision
    overall_confidence = mean(smoothed)
    is_fake = overall_confidence > 0.5
    
    return {
        'is_fake': is_fake,
        'confidence': overall_confidence,
        'segments': segments
    }
```

#### Temporal Smoothing
```python
def sliding_window_smoothing(predictions, window_size=5):
    # Apply moving average to reduce noise
    # Prevents sudden spikes from affecting decision
    smoothed = []
    for i in range(len(predictions)):
        start = max(0, i - window_size // 2)
        end = min(len(predictions), i + window_size // 2 + 1)
        avg = mean(predictions[start:end])
        smoothed.append(avg)
    return smoothed
```

#### Segment Detection
```python
def identify_manipulated_segments(predictions, timestamps, threshold=0.7):
    # Find consecutive high-confidence regions
    segments = []
    in_segment = False
    
    for i, (pred, (start_t, end_t)) in enumerate(zip(predictions, timestamps)):
        if pred > threshold and not in_segment:
            # Start new segment
            segment_start = start_t
            in_segment = True
        elif pred <= threshold and in_segment:
            # End current segment
            segments.append({
                'start_time': format_timestamp(segment_start),
                'end_time': format_timestamp(end_t),
                'confidence': max(predictions[...])
            })
            in_segment = False
    
    return segments
```

### 2. Batch Inference (`batch_inference.py`)

**Purpose**: Process multiple preprocessed videos efficiently for evaluation.

#### Architecture
```python
def batch_predict_preprocessed(model, data_dir, split='test', device='cuda'):
    model.eval()
    
    # Structure: data_dir/split/category/*.pt
    categories = ['Celeb-real', 'YouTube-real', 'Celeb-synthesis']
    
    results = {}
    
    for category in categories:
        pt_files = glob(f"{data_dir}/{split}/{category}/*.pt")
        
        for pt_file in tqdm(pt_files):
            # Load preprocessed windows
            data = torch.load(pt_file)
            windows = data['windows']
            
            # Predict all windows
            predictions = []
            for window in windows:
                with torch.no_grad():
                    logit = model(window)
                    prob = sigmoid(logit)
                    predictions.append(prob.item())
            
            # Aggregate video-level prediction
            avg_confidence = np.mean(predictions)
            is_fake = avg_confidence > 0.5
            
            results[video_name] = {
                'predicted': is_fake,
                'confidence': avg_confidence,
                'true_label': get_label(category),
                'num_windows': len(predictions)
            }
    
    return results
```

---

## Evaluation Framework

### 1. Comprehensive Metrics (`evaluate.py`)

**Purpose**: Calculate standard classification metrics and generate visualizations.

#### Metrics Computed
```python
Metrics:
  - Accuracy: (TP + TN) / Total
  - Precision: TP / (TP + FP)
  - Recall: TP / (TP + FN)
  - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
  - AUC-ROC: Area under ROC curve
  - Confusion Matrix: [[TN, FP], [FN, TP]]

Visualizations:
  - ROC Curve (TPR vs FPR)
  - Confusion Matrix heatmap
  - Confidence Distribution histogram
```

### 2. Segment-Level Evaluation (`evaluate_new.py`)

**Purpose**: Combine overall metrics with detailed temporal segment predictions.

#### Output Format
```json
{
  "overall_metrics": {
    "accuracy": 0.92,
    "precision": 0.90,
    "recall": 0.94,
    "f1_score": 0.92,
    "auc_roc": 0.96,
    "confusion_matrix": [[45, 5], [3, 47]]
  },
  "per_video_predictions": [
    {
      "video_name": "id0_0001",
      "input_type": "preprocessed",
      "video_is_fake": true,
      "overall_confidence": 0.87,
      "manipulated_segments": [
        {
          "start_time": "00:00:02",
          "end_time": "00:00:08",
          "confidence": 0.92
        }
      ]
    }
  ]
}
```

---

## Key Design Decisions

### 1. Temporal Windowing
**Decision**: 30 frames (10 seconds) with 2-second overlap  
**Rationale**: Balance between temporal context and computational efficiency; overlap prevents boundary artifacts.

### 2. Hybrid CNN-RNN Architecture
**Decision**: ResNeXt50 + Bidirectional GRU  
**Rationale**: ResNeXt provides strong spatial features; GRU captures temporal inconsistencies; bidirectional captures context in both directions.

### 3. BCEWithLogitsLoss
**Decision**: Use BCEWithLogitsLoss instead of BCE + Sigmoid  
**Rationale**: Numerically stable; supports mixed precision; fuses operations for efficiency.

### 4. Mixed Precision Training
**Decision**: AMP with FP16  
**Rationale**: 2x faster training; reduces memory usage; maintains accuracy with gradient scaling.

### 5. Preprocessing Separation
**Decision**: Separate preprocessing from training  
**Rationale**: Process once, train many times; faster iteration during model development; easier debugging.

### 6. Multi-Level Evaluation
**Decision**: Three evaluation scripts (metrics, batch, segments)  
**Rationale**: Different use cases - overall performance, batch processing, detailed analysis.

---

## Performance Characteristics

### Training Speed (Estimated)
```
Configuration: ResNeXt50+GRU, batch_size=4, AMP enabled

Without Optimizations:
  - Time per epoch: ~20-30 minutes
  - Total training (10 epochs): ~4-5 hours

With Optimizations (freeze_cnn=True, batch_size=8):
  - Time per epoch: ~6-8 minutes
  - Total training (10 epochs): ~1-1.5 hours
```

### Memory Requirements
```
GPU Memory (batch_size=4):
  - Model: ~500 MB
  - Batch: ~1.5 GB
  - Total: ~2-2.5 GB (fits on most modern GPUs)

Disk Space:
  - Raw videos: ~50-100 GB
  - Preprocessed .pt files: ~10-20 GB
  - Model checkpoints: ~500 MB per checkpoint
```

### Inference Speed
```
Single Video (30 seconds):
  - Preprocessing: ~5-10 seconds
  - Inference: ~1-2 seconds
  - Total: ~6-12 seconds

Batch Inference (100 videos):
  - Preprocessed: ~2-3 minutes
  - Raw videos: ~15-20 minutes (includes preprocessing)
```

---

## Technology Stack

### Core Libraries
```
PyTorch 2.x         - Deep learning framework
torchvision         - Pretrained models (ResNeXt50)
facenet-pytorch     - MTCNN face detection
opencv-python       - Video processing
Pillow              - Image handling
numpy               - Numerical operations
scikit-learn        - Metrics (ROC, confusion matrix)
tqdm                - Progress bars
matplotlib          - Visualization
```

### Hardware Requirements
```
Minimum:
  - CPU: 4+ cores
  - RAM: 16 GB
  - GPU: 4 GB VRAM (GTX 1650 or better)
  - Storage: 100 GB

Recommended:
  - CPU: 8+ cores
  - RAM: 32 GB
  - GPU: 8+ GB VRAM (RTX 3060 or better)
  - Storage: 200+ GB SSD
```

---

## Conclusion

This architecture provides a robust, efficient, and scalable deepfake detection system that:

1. **Preprocesses videos efficiently** with face detection and temporal windowing
2. **Leverages transfer learning** with ResNeXt50 pretrained features
3. **Captures temporal dynamics** using bidirectional GRU
4. **Trains efficiently** with mixed precision and optimization techniques
5. **Provides comprehensive evaluation** with metrics and segment-level analysis
6. **Scales to production** with batch inference and modular design

The modular design allows for easy experimentation with different components (e.g., different CNN backbones, RNN architectures, or training strategies) while maintaining a clean separation of concerns.
