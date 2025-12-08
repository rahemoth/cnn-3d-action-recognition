"""Inference module for action recognition."""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
import torch

from src.models.cnn2d import CNN2D
from src.models.cnn3d import CNN3D
from src.training.config import InferenceConfig


class ActionPredictor:
    """Predictor for action recognition from video."""
    
    def __init__(self, config: InferenceConfig, class_names: List[str]):
        """
        Args:
            config: Inference configuration
            class_names: List of action class names
        """
        self.config = config
        self.class_names = class_names
        self.device = torch.device(config.device)
        
        # Initialize model
        if config.model_type == "cnn2d":
            self.model = CNN2D(
                num_classes=config.num_classes,
                num_frames=config.num_frames
            )
        elif config.model_type == "cnn3d":
            self.model = CNN3D(num_classes=config.num_classes)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load checkpoint if available
        if Path(config.checkpoint_path).exists():
            self.load_checkpoint(config.checkpoint_path)
        else:
            print(f"Warning: Checkpoint not found at {config.checkpoint_path}")
            print("Using randomly initialized model.")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    
    def preprocess_video(self, video_path: str) -> torch.Tensor:
        """
        Load and preprocess video from file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Preprocessed frames tensor of shape (1, num_frames, C, H, W)
        """
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize
            frame = cv2.resize(frame, (self.config.image_size[1], self.config.image_size[0]))
            
            frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        if frame_count == 0:
            raise ValueError(f"No frames loaded from {video_path}")
        
        # Sample frames uniformly
        frames = self._sample_frames(frames, self.config.num_frames)
        
        # Convert to tensor
        frames = np.array(frames).astype(np.float32) / 255.0
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        frames = frames.unsqueeze(0)  # Add batch dimension
        
        return frames
    
    def preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess a list of frame arrays.
        
        Args:
            frames: List of frames as numpy arrays (H, W, C)
            
        Returns:
            Preprocessed frames tensor of shape (1, num_frames, C, H, W)
        """
        processed_frames = []
        
        for frame in frames:
            # Resize
            frame = cv2.resize(frame, (self.config.image_size[1], self.config.image_size[0]))
            processed_frames.append(frame)
        
        # Sample frames
        processed_frames = self._sample_frames(processed_frames, self.config.num_frames)
        
        # Convert to tensor
        processed_frames = np.array(processed_frames).astype(np.float32) / 255.0
        processed_frames = torch.from_numpy(processed_frames).permute(0, 3, 1, 2)
        processed_frames = processed_frames.unsqueeze(0)
        
        return processed_frames
    
    def _sample_frames(self, frames: List[np.ndarray], num_frames: int) -> List[np.ndarray]:
        """Sample frames uniformly from video."""
        total_frames = len(frames)
        
        if total_frames >= num_frames:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            return [frames[i] for i in indices]
        else:
            # Repeat frames if not enough
            indices = np.arange(total_frames)
            indices = np.pad(indices, (0, num_frames - total_frames), mode='edge')
            return [frames[i] for i in indices]
    
    def predict(self, frames: torch.Tensor) -> Tuple[str, float, np.ndarray]:
        """
        Predict action from frames.
        
        Args:
            frames: Preprocessed frames tensor
            
        Returns:
            Tuple of (predicted_class_name, confidence, probabilities)
        """
        frames = frames.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(frames)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = probabilities.max(1)
        
        predicted_class = self.class_names[predicted_idx.item()]
        confidence = confidence.item()
        probs = probabilities.cpu().numpy()[0]
        
        return predicted_class, confidence, probs
    
    def predict_from_video(self, video_path: str) -> Tuple[str, float, np.ndarray]:
        """
        Predict action directly from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (predicted_class_name, confidence, probabilities)
        """
        frames = self.preprocess_video(video_path)
        return self.predict(frames)
