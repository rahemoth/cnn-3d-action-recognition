"""Pose interface for skeleton representation."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class PoseKeypoint:
    """Single keypoint in 2D or 3D space."""
    x: float
    y: float
    z: float = 0.0  # Optional for 2D poses
    confidence: float = 1.0
    name: str = ""


@dataclass
class PoseSequence:
    """
    Sequence of poses over time.
    
    Represents a temporal sequence of body poses, suitable for:
    - Driving 3D skeleton animations
    - Action recognition visualization
    - Pose-based action analysis
    """
    
    keypoints: List[List[PoseKeypoint]]  # List of frames, each containing list of keypoints
    fps: float = 30.0
    action_label: Optional[str] = None
    
    def __len__(self) -> int:
        """Return number of frames."""
        return len(self.keypoints)
    
    def get_frame(self, idx: int) -> List[PoseKeypoint]:
        """Get keypoints for a specific frame."""
        return self.keypoints[idx]
    
    def to_array(self) -> np.ndarray:
        """
        Convert to numpy array.
        
        Returns:
            Array of shape (num_frames, num_keypoints, 3) with x, y, z coordinates
        """
        num_frames = len(self.keypoints)
        if num_frames == 0:
            return np.array([])
        
        num_keypoints = len(self.keypoints[0])
        arr = np.zeros((num_frames, num_keypoints, 3))
        
        for frame_idx, frame in enumerate(self.keypoints):
            for kp_idx, kp in enumerate(frame):
                arr[frame_idx, kp_idx] = [kp.x, kp.y, kp.z]
        
        return arr


# Define standard skeleton connections (body edges)
SKELETON_CONNECTIONS = [
    # Head to torso
    (0, 1),   # Nose to Neck
    
    # Arms
    (1, 2),   # Neck to Right Shoulder
    (2, 3),   # Right Shoulder to Right Elbow
    (3, 4),   # Right Elbow to Right Wrist
    (1, 5),   # Neck to Left Shoulder
    (5, 6),   # Left Shoulder to Left Elbow
    (6, 7),   # Left Elbow to Left Wrist
    
    # Torso
    (1, 8),   # Neck to Mid Hip
    
    # Legs
    (8, 9),   # Mid Hip to Right Hip
    (9, 10),  # Right Hip to Right Knee
    (10, 11), # Right Knee to Right Ankle
    (8, 12),  # Mid Hip to Left Hip
    (12, 13), # Left Hip to Left Knee
    (13, 14), # Left Knee to Left Ankle
]

KEYPOINT_NAMES = [
    "Nose",           # 0
    "Neck",           # 1
    "RShoulder",      # 2
    "RElbow",         # 3
    "RWrist",         # 4
    "LShoulder",      # 5
    "LElbow",         # 6
    "LWrist",         # 7
    "MidHip",         # 8
    "RHip",           # 9
    "RKnee",          # 10
    "RAnkle",         # 11
    "LHip",           # 12
    "LKnee",          # 13
    "LAnkle",         # 14
]


def generate_mock_pose_sequence(
    action_label: str,
    num_frames: int = 30,
    fps: float = 30.0
) -> PoseSequence:
    """
    Generate a mock pose sequence for demonstration.
    
    Creates synthetic pose keypoints that vary over time to simulate
    different actions. This is a placeholder until a real pose estimator
    is integrated.
    
    Args:
        action_label: Name of the action (affects motion pattern)
        num_frames: Number of frames to generate
        fps: Frame rate
        
    Returns:
        PoseSequence with synthetic keypoints
    """
    keypoints_sequence = []
    num_keypoints = len(KEYPOINT_NAMES)
    
    # Define base pose (T-pose normalized coordinates)
    base_pose = {
        0: (0.0, 1.8, 0.0),    # Nose
        1: (0.0, 1.5, 0.0),    # Neck
        2: (0.3, 1.4, 0.0),    # RShoulder
        3: (0.5, 1.2, 0.0),    # RElbow
        4: (0.7, 1.0, 0.0),    # RWrist
        5: (-0.3, 1.4, 0.0),   # LShoulder
        6: (-0.5, 1.2, 0.0),   # LElbow
        7: (-0.7, 1.0, 0.0),   # LWrist
        8: (0.0, 1.0, 0.0),    # MidHip
        9: (0.2, 1.0, 0.0),    # RHip
        10: (0.2, 0.5, 0.0),   # RKnee
        11: (0.2, 0.0, 0.0),   # RAnkle
        12: (-0.2, 1.0, 0.0),  # LHip
        13: (-0.2, 0.5, 0.0),  # LKnee
        14: (-0.2, 0.0, 0.0),  # LAnkle
    }
    
    for frame_idx in range(num_frames):
        t = frame_idx / num_frames
        frame_keypoints = []
        
        for kp_idx in range(num_keypoints):
            base_x, base_y, base_z = base_pose[kp_idx]
            
            # Add action-specific motion
            if action_label.lower() == "walking":
                # Simulate walking motion
                leg_swing = 0.3 * np.sin(2 * np.pi * t * 2)
                if kp_idx in [9, 10, 11]:  # Right leg
                    base_z += leg_swing
                elif kp_idx in [12, 13, 14]:  # Left leg
                    base_z -= leg_swing
                # Arm swing
                arm_swing = 0.2 * np.sin(2 * np.pi * t * 2)
                if kp_idx in [2, 3, 4]:  # Right arm
                    base_z -= arm_swing
                elif kp_idx in [5, 6, 7]:  # Left arm
                    base_z += arm_swing
                    
            elif action_label.lower() == "running":
                # Simulate running motion (faster and more pronounced)
                leg_swing = 0.5 * np.sin(2 * np.pi * t * 4)
                if kp_idx in [9, 10, 11]:  # Right leg
                    base_z += leg_swing
                elif kp_idx in [12, 13, 14]:  # Left leg
                    base_z -= leg_swing
                # More arm swing
                arm_swing = 0.4 * np.sin(2 * np.pi * t * 4)
                if kp_idx in [2, 3, 4]:  # Right arm
                    base_z -= arm_swing
                elif kp_idx in [5, 6, 7]:  # Left arm
                    base_z += arm_swing
                    
            elif action_label.lower() == "jumping":
                # Simulate jumping motion
                jump_height = 0.5 * np.abs(np.sin(np.pi * t))
                base_y += jump_height
                # Arms up during jump
                if kp_idx in [4, 7]:  # Wrists
                    base_y += 0.3 * np.abs(np.sin(np.pi * t))
            
            # Add small random noise for realism
            noise_scale = 0.01
            x = base_x + np.random.randn() * noise_scale
            y = base_y + np.random.randn() * noise_scale
            z = base_z + np.random.randn() * noise_scale
            
            keypoint = PoseKeypoint(
                x=x, y=y, z=z,
                confidence=0.95,
                name=KEYPOINT_NAMES[kp_idx]
            )
            frame_keypoints.append(keypoint)
        
        keypoints_sequence.append(frame_keypoints)
    
    return PoseSequence(
        keypoints=keypoints_sequence,
        fps=fps,
        action_label=action_label
    )
