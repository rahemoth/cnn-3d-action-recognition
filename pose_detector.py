"""
Pose detection module using MediaPipe and OpenCV.
Detects human body keypoints from video input.
"""

import cv2
import mediapipe as mp
import numpy as np


class PoseDetector:
    """Detects human pose keypoints using MediaPipe."""
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the pose detector.
        
        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,
            enable_segmentation=False
        )
        
    def detect(self, frame):
        """
        Detect pose landmarks in a frame.
        
        Args:
            frame: Input image (BGR format from OpenCV)
            
        Returns:
            results: MediaPipe pose detection results
            rgb_frame: RGB converted frame
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        return results, rgb_frame
    
    def extract_keypoints(self, results, image_shape):
        """
        Extract 3D keypoints from detection results.
        
        Args:
            results: MediaPipe pose detection results
            image_shape: Shape of the input image (height, width, channels)
            
        Returns:
            keypoints_3d: Array of shape (33, 4) containing x, y, z, visibility
                         Returns None if no pose detected
        """
        if not results.pose_landmarks:
            return None
        
        landmarks = results.pose_landmarks.landmark
        height, width = image_shape[:2]
        
        # Extract keypoints with normalized coordinates
        keypoints_3d = []
        for landmark in landmarks:
            # x, y are normalized [0, 1], z is depth relative to hips
            # visibility is confidence [0, 1]
            keypoints_3d.append([
                landmark.x * width,
                landmark.y * height,
                landmark.z * width,  # Scale z by width for better visualization
                landmark.visibility
            ])
        
        return np.array(keypoints_3d)
    
    def draw_landmarks(self, frame, results):
        """
        Draw pose landmarks on the frame.
        
        Args:
            frame: Input frame (BGR format)
            results: MediaPipe pose detection results
            
        Returns:
            annotated_frame: Frame with drawn landmarks
        """
        annotated_frame = frame.copy()
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return annotated_frame
    
    def get_key_body_points(self, keypoints_3d):
        """
        Extract key body endpoints for simplified 3D visualization.
        
        Args:
            keypoints_3d: Array of all keypoints (33, 4)
            
        Returns:
            key_points: Dictionary of key body endpoints
        """
        if keypoints_3d is None:
            return None
        
        # MediaPipe pose landmark indices
        key_indices = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
        }
        
        key_points = {}
        for name, idx in key_indices.items():
            key_points[name] = keypoints_3d[idx][:3]  # x, y, z only
        
        return key_points
    
    def close(self):
        """Release resources."""
        self.pose.close()
