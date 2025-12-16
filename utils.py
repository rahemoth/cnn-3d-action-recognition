"""
Utility functions for the 3D body motion recognition system.
"""

import numpy as np


def calculate_angle(point1, point2, point3):
    """
    Calculate angle between three points.
    
    Args:
        point1, point2, point3: 3D points as numpy arrays
        
    Returns:
        angle: Angle in degrees
    """
    vector1 = point1 - point2
    vector2 = point3 - point2
    
    # Calculate angle
    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    return np.degrees(angle)


def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two 3D points.
    
    Args:
        point1, point2: 3D points as numpy arrays
        
    Returns:
        distance: Euclidean distance
    """
    return np.linalg.norm(point1 - point2)


def normalize_skeleton(key_points):
    """
    Normalize skeleton to unit scale based on torso height.
    
    Args:
        key_points: Dictionary of key body points
        
    Returns:
        normalized_points: Normalized key points
    """
    if key_points is None:
        return None
    
    # Calculate torso height (shoulder to hip)
    left_shoulder = key_points.get('left_shoulder')
    left_hip = key_points.get('left_hip')
    
    if left_shoulder is None or left_hip is None:
        return key_points
    
    torso_height = calculate_distance(left_shoulder, left_hip)
    
    if torso_height < 1e-6:  # Avoid division by zero
        return key_points
    
    # Normalize all points
    normalized_points = {}
    for name, point in key_points.items():
        normalized_points[name] = point / torso_height
    
    return normalized_points


def get_body_center(key_points):
    """
    Calculate the center of the body (average of hips).
    
    Args:
        key_points: Dictionary of key body points
        
    Returns:
        center: 3D center point
    """
    if key_points is None:
        return None
    
    left_hip = key_points.get('left_hip')
    right_hip = key_points.get('right_hip')
    
    if left_hip is None or right_hip is None:
        return None
    
    return (left_hip + right_hip) / 2.0


def filter_keypoints(keypoints_3d, threshold=0.5):
    """
    Filter out keypoints with low visibility.
    
    Args:
        keypoints_3d: Array of keypoints with visibility scores
        threshold: Minimum visibility threshold
        
    Returns:
        filtered_keypoints: Keypoints with visibility above threshold
    """
    if keypoints_3d is None:
        return None
    
    # Create a mask for visible keypoints
    visibility_mask = keypoints_3d[:, 3] >= threshold
    
    # Keep only visible keypoints
    filtered = keypoints_3d.copy()
    filtered[~visibility_mask] = np.nan
    
    return filtered
