#!/usr/bin/env python3
"""Demo inference script for action recognition."""

import os
import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2

from src.inference.predictor import ActionPredictor
from src.training.config import InferenceConfig
from src.visualization.pose_interface import generate_mock_pose_sequence
from src.visualization.skeleton_3d_demo import Skeleton3DVisualizer


def create_sample_video(output_path: str, action: str = "walking", duration: int = 2):
    """
    Create a simple sample video for testing.
    
    Args:
        output_path: Path to save video
        action: Action type (affects color)
        duration: Video duration in seconds
    """
    fps = 10
    num_frames = fps * duration
    height, width = 112, 112
    
    # Define color based on action (matching dataset generation)
    if action == "walking":
        base_color = (200, 150, 100)  # BGR
    elif action == "running":
        base_color = (150, 100, 200)
    else:  # jumping
        base_color = (100, 200, 150)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(num_frames):
        # Create frame with temporal variation
        color_offset = int(20 * np.sin(i / num_frames * 2 * np.pi))
        frame = np.ones((height, width, 3), dtype=np.uint8) * np.array(base_color)
        frame = np.clip(frame + color_offset, 0, 255).astype(np.uint8)
        
        # Add noise
        noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        out.write(frame)
    
    out.release()
    print(f"Created sample video: {output_path}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run inference on a video")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input video file (if not provided, uses sample video)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show 3D skeleton visualization"
    )
    
    args = parser.parse_args()
    
    print("CNN Action Recognition - Inference Demo")
    print("=" * 60)
    
    # Configuration
    config = InferenceConfig(
        model_type="cnn2d",
        num_classes=3,
        num_frames=8,
        image_size=(112, 112),
        checkpoint_path=args.checkpoint,
        device="cpu"
    )
    
    # Define class names (should match training)
    class_names = ["walking", "running", "jumping"]
    
    # Initialize predictor
    print("\nInitializing predictor...")
    predictor = ActionPredictor(config, class_names)
    
    # Get or create video
    if args.video is None:
        # Create a sample video
        sample_dir = Path("tmp")
        sample_dir.mkdir(exist_ok=True)
        video_path = str(sample_dir / "sample_walking.mp4")
        
        if not Path(video_path).exists():
            print("\nCreating sample video...")
            create_sample_video(video_path, action="walking", duration=2)
        else:
            print(f"\nUsing existing sample video: {video_path}")
    else:
        video_path = args.video
        print(f"\nUsing video: {video_path}")
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        return
    
    # Run inference
    print("\nRunning inference...")
    try:
        predicted_class, confidence, probabilities = predictor.predict_from_video(video_path)
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Predicted Action: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        print("\nClass Probabilities:")
        for class_name, prob in zip(class_names, probabilities):
            print(f"  {class_name:15s}: {prob:.2%}")
        print("=" * 60)
        
        # Visualize 3D skeleton if requested
        if args.visualize:
            print("\nGenerating 3D skeleton visualization...")
            pose_seq = generate_mock_pose_sequence(
                action_label=predicted_class,
                num_frames=30,
                fps=30.0
            )
            
            print("Showing 3D skeleton animation (close window to exit)...")
            visualizer = Skeleton3DVisualizer()
            visualizer.animate(pose_seq, interval=33)
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nInference demo completed!")


if __name__ == "__main__":
    main()
