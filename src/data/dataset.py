"""Dataset loader for video action recognition."""

import os
from pathlib import Path
from typing import Tuple, List
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class VideoFrameDataset(Dataset):
    """
    Dataset for loading video frames organized by action class.
    
    Expected directory structure:
        data_dir/
            action_class_1/
                video_001/
                    frame_0001.jpg
                    frame_0002.jpg
                    ...
                video_002/
                    ...
            action_class_2/
                ...
    """
    
    def __init__(
        self, 
        data_dir: str, 
        num_frames: int = 8,
        image_size: Tuple[int, int] = (112, 112),
        transform=None
    ):
        """
        Args:
            data_dir: Root directory containing action class folders
            num_frames: Number of frames to sample from each video
            image_size: Target size for frames (height, width)
            transform: Optional transforms to apply
        """
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.image_size = image_size
        self.transform = transform
        
        # Build dataset index
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        if self.data_dir.exists():
            self._build_dataset()
    
    def _build_dataset(self):
        """Scan directory structure and build sample list."""
        # Get action classes
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Scan for video folders
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            video_dirs = sorted([d for d in class_dir.iterdir() if d.is_dir()])
            
            for video_dir in video_dirs:
                # Get frame files
                frames = sorted(list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png")))
                if len(frames) >= self.num_frames:
                    self.samples.append({
                        'video_path': video_dir,
                        'class_idx': class_idx,
                        'class_name': class_dir.name,
                        'num_frames': len(frames)
                    })
    
    def _load_frames(self, video_path: Path, num_available_frames: int) -> np.ndarray:
        """
        Load and sample frames from video directory.
        
        Returns:
            Array of shape (num_frames, height, width, channels)
        """
        # Sample frame indices uniformly
        if num_available_frames > self.num_frames:
            indices = np.linspace(0, num_available_frames - 1, self.num_frames, dtype=int)
        else:
            # If fewer frames than needed, repeat frames
            indices = np.arange(num_available_frames)
            indices = np.pad(indices, (0, self.num_frames - num_available_frames), mode='edge')
        
        frames = []
        frame_files = sorted(list(video_path.glob("*.jpg")) + list(video_path.glob("*.png")))
        
        for idx in indices:
            frame_path = frame_files[idx]
            frame = cv2.imread(str(frame_path))
            if frame is None:
                # Create blank frame if loading fails
                frame = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
            frames.append(frame)
        
        return np.array(frames)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a video sample.
        
        Returns:
            frames: Tensor of shape (num_frames, C, H, W) or (C*num_frames, H, W) for 2D CNN
            label: Class index
        """
        sample = self.samples[idx]
        
        # Load frames
        frames = self._load_frames(sample['video_path'], sample['num_frames'])
        
        # Normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Convert to tensor: (T, H, W, C) -> (T, C, H, W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        
        if self.transform:
            frames = self.transform(frames)
        
        label = sample['class_idx']
        
        return frames, label


def create_sample_dataset(data_dir: str = "data/train", num_classes: int = 3):
    """
    Create a tiny sample dataset for testing purposes.
    Generates synthetic video frames for testing.
    
    Args:
        data_dir: Directory to create sample data
        num_classes: Number of action classes to generate
    """
    data_path = Path(data_dir)
    class_names = ['walking', 'running', 'jumping'][:num_classes]
    
    print(f"Creating sample dataset at {data_dir}...")
    
    for class_name in class_names:
        # Create 2 videos per class
        for video_idx in range(2):
            video_dir = data_path / class_name / f"video_{video_idx:03d}"
            video_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate 16 synthetic frames per video
            for frame_idx in range(16):
                # Create a simple synthetic frame with random noise
                # Each class has a different base color for easy visual distinction
                if class_name == 'walking':
                    base_color = np.array([100, 150, 200])
                elif class_name == 'running':
                    base_color = np.array([200, 100, 150])
                else:  # jumping
                    base_color = np.array([150, 200, 100])
                
                # Add some temporal variation
                color_offset = int(20 * np.sin(frame_idx / 16.0 * 2 * np.pi))
                frame = np.ones((112, 112, 3), dtype=np.uint8) * base_color
                frame = np.clip(frame + color_offset, 0, 255).astype(np.uint8)
                
                # Add some random noise
                noise = np.random.randint(-20, 20, (112, 112, 3), dtype=np.int16)
                frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                # Save frame
                frame_path = video_dir / f"frame_{frame_idx:04d}.jpg"
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    print(f"Sample dataset created with {num_classes} classes, 2 videos per class.")
