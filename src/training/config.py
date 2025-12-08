"""Configuration settings for training and inference."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Model settings
    model_type: str = "cnn2d"  # Options: "cnn2d" or "cnn3d"
    num_classes: int = 3
    
    # Data settings
    num_frames: int = 8
    image_size: Tuple[int, int] = (112, 112)
    fps: int = 10
    
    # Training settings
    batch_size: int = 4
    num_epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Data paths
    data_dir: str = "data/train"
    val_dir: str = "data/val"
    checkpoint_dir: str = "checkpoints"
    
    # Device
    device: str = "cpu"  # CPU-friendly default
    
    # Logging
    log_interval: int = 10
    save_interval: int = 5


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    
    model_type: str = "cnn2d"
    num_classes: int = 3
    num_frames: int = 8
    image_size: Tuple[int, int] = (112, 112)
    fps: int = 10
    checkpoint_path: str = "checkpoints/best_model.pth"
    device: str = "cpu"
