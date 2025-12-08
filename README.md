# CNN-based 3D Action Recognition

A PyTorch-based project for human action recognition from video using CNN architectures, with 3D skeleton visualization capabilities.

## Overview

This project provides a complete pipeline for:
- **Action Recognition**: CNN-based models (2D and 3D) for classifying actions from video sequences
- **3D Visualization**: Animated 3D skeleton visualization driven by pose keypoints
- **Easy Experimentation**: Simple APIs and scripts for training and inference

The implementation focuses on clean, readable code with CPU-friendly defaults, making it perfect for learning and prototyping on typical laptops.

## Features

- **Multiple CNN Architectures**:
  - 2D CNN with temporal aggregation (stacked frames)
  - Lightweight 3D CNN for spatiotemporal feature learning
- **Flexible Data Pipeline**: Load videos from organized directory structures
- **Training Framework**: Complete training loop with validation and checkpointing
- **Inference API**: Easy-to-use predictor for single videos or frame sequences
- **3D Skeleton Visualization**: Matplotlib-based animated skeleton rendering
- **Pose Interface**: Extensible framework for integrating real pose estimators (MediaPipe, OpenPose, etc.)

## Project Structure

```
cnn-3d-action-recognition/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py          # Video frame dataset loader
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn2d.py            # 2D CNN architecture
│   │   └── cnn3d.py            # 3D CNN architecture
│   ├── training/
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration dataclasses
│   │   └── trainer.py          # Training loop
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py        # Inference API
│   └── visualization/
│       ├── __init__.py
│       ├── pose_interface.py   # Pose data structures
│       └── skeleton_3d_demo.py # 3D skeleton renderer
├── scripts/
│   ├── train_example.py        # Training example
│   └── demo_inference.py       # Inference demo
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rahemoth/cnn-3d-action-recognition.git
   cd cnn-3d-action-recognition
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # On Linux/macOS
   python -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Training

Train a model on a sample dataset:

```bash
python scripts/train_example.py
```

This will:
- Automatically generate a small synthetic dataset (if not present)
- Train a 2D CNN for 10 epochs
- Save checkpoints to `checkpoints/`
- Display training and validation metrics

**Sample Output**:
```
CNN Action Recognition - Training Example
============================================================
Creating sample training dataset at data/train...
Sample dataset created with 3 classes, 2 videos per class.
...
Epoch 10/10
Train Loss: 0.2345
Val Loss: 0.3456, Val Acc: 85.00%
Training completed!
Best validation accuracy: 85.00%
```

### 2. Inference

Run inference on a video:

```bash
# Use auto-generated sample video
python scripts/demo_inference.py

# Use custom video
python scripts/demo_inference.py --video path/to/your/video.mp4

# With 3D skeleton visualization
python scripts/demo_inference.py --visualize
```

**Sample Output**:
```
CNN Action Recognition - Inference Demo
============================================================
Running inference...

============================================================
RESULTS
============================================================
Predicted Action: walking
Confidence: 92.34%

Class Probabilities:
  walking        : 92.34%
  running        : 5.43%
  jumping        : 2.23%
============================================================
```

### 3. 3D Skeleton Visualization

Run standalone 3D skeleton demo:

```bash
python src/visualization/skeleton_3d_demo.py
```

This demonstrates animated 3D skeleton visualization for walking, running, and jumping actions.

## Data Format

The project expects videos organized by action class:

```
data/
├── train/
│   ├── walking/
│   │   ├── video_001/
│   │   │   ├── frame_0001.jpg
│   │   │   ├── frame_0002.jpg
│   │   │   └── ...
│   │   ├── video_002/
│   │   └── ...
│   ├── running/
│   └── jumping/
└── val/
    └── (same structure)
```

**Supported formats**: `.jpg`, `.png` frames or `.mp4`, `.avi` video files

### Preparing Your Own Dataset

1. **Organize videos** into class folders as shown above
2. **Extract frames** from videos (optional):
   ```bash
   ffmpeg -i video.mp4 -vf fps=10 frame_%04d.jpg
   ```
3. **Update configuration** in `scripts/train_example.py`:
   ```python
   config = TrainingConfig(
       num_classes=5,  # Your number of classes
       data_dir="path/to/your/train",
       val_dir="path/to/your/val",
       ...
   )
   ```

## Configuration

Key hyperparameters in `src/training/config.py`:

```python
@dataclass
class TrainingConfig:
    model_type: str = "cnn2d"           # "cnn2d" or "cnn3d"
    num_classes: int = 3                # Number of action classes
    num_frames: int = 8                 # Frames per video clip
    image_size: Tuple[int, int] = (112, 112)  # (height, width)
    batch_size: int = 4                 # Batch size
    num_epochs: int = 20                # Training epochs
    learning_rate: float = 0.001        # Learning rate
    device: str = "cpu"                 # "cpu" or "cuda"
```

## Model Architectures

### 2D CNN (Default)
- Treats temporal dimension by stacking frames as input channels
- Input: `(batch, num_frames * 3, height, width)`
- Lighter weight, faster training
- Good for simple actions with limited temporal dependencies

### 3D CNN
- Processes video with explicit temporal convolutions
- Input: `(batch, 3, num_frames, height, width)`
- Better at capturing temporal motion patterns
- More parameters, slower training

## Extending the Project

### Adding Real Pose Estimation

The pose interface is designed to integrate with real pose estimators:

```python
from src.visualization.pose_interface import PoseSequence, PoseKeypoint

# Example with MediaPipe (install mediapipe first)
import mediapipe as mp

def extract_poses_from_video(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    
    # Process video and extract keypoints
    # Convert to PoseSequence format
    # ...
    
    return PoseSequence(keypoints=keypoints_sequence, fps=30.0)
```

### Adding New Models

1. Create a new model file in `src/models/`
2. Inherit from `nn.Module`
3. Update `src/training/trainer.py` to support the new model type

### Custom Data Augmentation

Add transforms in the dataset loader:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
    # Add more augmentations
])

dataset = VideoFrameDataset(
    data_dir="data/train",
    transform=transform
)
```

## Performance Tips

### For CPU Training
- Use smaller batch sizes (2-4)
- Reduce image size (e.g., 64x64 or 112x112)
- Use fewer frames (4-8)
- Start with 2D CNN (faster than 3D)

### For GPU Training
- Update `device` to `"cuda"` in config
- Increase batch size (16-32)
- Use larger image sizes (224x224)
- Consider deeper architectures

## Troubleshooting

### ImportError or ModuleNotFoundError
Make sure you're running scripts from the project root:
```bash
cd cnn-3d-action-recognition
python scripts/train_example.py
```

### Out of Memory
- Reduce `batch_size` in config
- Reduce `image_size` or `num_frames`
- Close other applications

### Slow Training
- This is expected on CPU for deep learning
- Use fewer epochs for quick testing
- Consider using a GPU or cloud service

## Future Enhancements

- [ ] Real-time webcam inference
- [ ] Integration with MediaPipe/OpenPose for pose estimation
- [ ] Pretrained model weights
- [ ] More advanced 3D visualization (Open3D, PyQt)
- [ ] Support for popular datasets (UCF101, HMDB51)
- [ ] Temporal attention mechanisms
- [ ] Multi-person action recognition

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- PyTorch framework for deep learning
- Matplotlib for visualization
- OpenCV for video processing

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{cnn3d-action-recognition,
  author = {rahemoth},
  title = {CNN-based 3D Action Recognition},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/rahemoth/cnn-3d-action-recognition}
}
```