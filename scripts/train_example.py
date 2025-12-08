#!/usr/bin/env python3
"""Example training script for action recognition."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import VideoFrameDataset, create_sample_dataset
from src.training.config import TrainingConfig
from src.training.trainer import Trainer


def main():
    """Main training function."""
    print("CNN Action Recognition - Training Example")
    print("=" * 60)
    
    # Configuration
    config = TrainingConfig(
        model_type="cnn2d",  # Options: "cnn2d" or "cnn3d"
        num_classes=3,
        num_frames=8,
        image_size=(112, 112),
        batch_size=2,  # Small batch size for CPU
        num_epochs=10,
        learning_rate=0.001,
        data_dir="data/train",
        val_dir="data/val",
        checkpoint_dir="checkpoints",
        device="cpu",
        log_interval=1,
        save_interval=5
    )
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_type}")
    print(f"  Classes: {config.num_classes}")
    print(f"  Frames: {config.num_frames}")
    print(f"  Image size: {config.image_size}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Device: {config.device}")
    
    # Create sample dataset if it doesn't exist
    train_path = Path(config.data_dir)
    val_path = Path(config.val_dir)
    
    if not train_path.exists():
        print(f"\nCreating sample training dataset at {config.data_dir}...")
        create_sample_dataset(config.data_dir, config.num_classes)
    
    if not val_path.exists():
        print(f"Creating sample validation dataset at {config.val_dir}...")
        create_sample_dataset(config.val_dir, config.num_classes)
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = VideoFrameDataset(
        data_dir=config.data_dir,
        num_frames=config.num_frames,
        image_size=config.image_size
    )
    
    val_dataset = VideoFrameDataset(
        data_dir=config.val_dir,
        num_frames=config.num_frames,
        image_size=config.image_size
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # Use 0 for CPU-only training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(config)
    
    # Train model
    print("\nStarting training...")
    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    print("\nTraining script completed!")
    print(f"Checkpoints saved in: {config.checkpoint_dir}")


if __name__ == "__main__":
    main()
