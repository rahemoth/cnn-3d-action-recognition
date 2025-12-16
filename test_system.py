"""
Test script for the 3D body motion recognition system.
Tests all modules without requiring a camera.
"""

import numpy as np
import sys
import cv2


def test_pose_detector():
    """Test PoseDetector module."""
    print("Testing PoseDetector module...")
    
    from pose_detector import PoseDetector
    
    # Create detector
    detector = PoseDetector()
    
    # Create a dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Test detection (should return no results for blank frame)
    results, rgb_frame = detector.detect(frame)
    assert rgb_frame.shape == frame.shape, "RGB frame shape mismatch"
    
    # Test keypoint extraction with no detection
    keypoints = detector.extract_keypoints(results, frame.shape)
    assert keypoints is None, "Should return None for no detection"
    
    # Test get_key_body_points with None
    key_points = detector.get_key_body_points(None)
    assert key_points is None, "Should return None for None input"
    
    # Clean up
    detector.close()
    
    print("✓ PoseDetector tests passed")


def test_skeleton_3d():
    """Test Skeleton3D module."""
    print("Testing Skeleton3D module...")
    
    from skeleton_3d import Skeleton3D
    
    # Create visualizer
    skeleton = Skeleton3D(figsize=(8, 6))
    
    # Test with None input
    skeleton.update(None)
    
    # Test with dummy key points
    dummy_points = {
        'nose': np.array([320, 240, 0]),
        'left_shoulder': np.array([300, 260, 0]),
        'right_shoulder': np.array([340, 260, 0]),
        'left_elbow': np.array([280, 300, 0]),
        'right_elbow': np.array([360, 300, 0]),
        'left_wrist': np.array([270, 340, 0]),
        'right_wrist': np.array([370, 340, 0]),
        'left_hip': np.array([300, 380, 0]),
        'right_hip': np.array([340, 380, 0]),
        'left_knee': np.array([290, 440, 0]),
        'right_knee': np.array([350, 440, 0]),
        'left_ankle': np.array([285, 500, 0]),
        'right_ankle': np.array([355, 500, 0]),
    }
    
    skeleton.update(dummy_points)
    
    # Clean up
    skeleton.close()
    
    print("✓ Skeleton3D tests passed")


def test_utils():
    """Test utility functions."""
    print("Testing utility functions...")
    
    from utils import (
        calculate_angle,
        calculate_distance,
        normalize_skeleton,
        get_body_center,
        filter_keypoints
    )
    
    # Test angle calculation
    p1 = np.array([1, 0, 0])
    p2 = np.array([0, 0, 0])
    p3 = np.array([0, 1, 0])
    angle = calculate_angle(p1, p2, p3)
    assert abs(angle - 90.0) < 1e-6, "Angle calculation failed"
    
    # Test distance calculation
    dist = calculate_distance(p1, p3)
    assert abs(dist - np.sqrt(2)) < 1e-6, "Distance calculation failed"
    
    # Test get_body_center
    key_points = {
        'left_hip': np.array([100, 200, 0]),
        'right_hip': np.array([200, 200, 0])
    }
    center = get_body_center(key_points)
    assert np.allclose(center, [150, 200, 0]), "Body center calculation failed"
    
    # Test normalize_skeleton
    key_points_full = {
        'left_shoulder': np.array([100, 100, 0]),
        'left_hip': np.array([100, 200, 0]),
        'right_shoulder': np.array([200, 100, 0])
    }
    normalized = normalize_skeleton(key_points_full)
    assert normalized is not None, "Normalization failed"
    
    # Test filter_keypoints
    keypoints_with_visibility = np.array([
        [100, 200, 0, 0.9],
        [150, 250, 0, 0.3],
        [200, 300, 0, 0.8]
    ])
    filtered = filter_keypoints(keypoints_with_visibility, threshold=0.5)
    assert not np.isnan(filtered[0, 0]), "First keypoint should be visible"
    assert np.isnan(filtered[1, 0]), "Second keypoint should be filtered out"
    assert not np.isnan(filtered[2, 0]), "Third keypoint should be visible"
    
    print("✓ Utility function tests passed")


def test_main_module():
    """Test main module imports."""
    print("Testing main module...")
    
    # Just test that it can be imported
    import main
    
    # Test BodyMotion3D can be instantiated
    # (We won't actually run it since we don't have a camera)
    app = main.BodyMotion3D(source=None)
    assert app is not None, "BodyMotion3D instantiation failed"
    
    print("✓ Main module tests passed")


def test_demo_module():
    """Test demo module imports."""
    print("Testing demo module...")
    
    # Just test that it can be imported
    import demo
    
    print("✓ Demo module tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running 3D Body Motion Recognition System Tests")
    print("=" * 60)
    print()
    
    try:
        test_pose_detector()
        test_skeleton_3d()
        test_utils()
        test_main_module()
        test_demo_module()
        
        print()
        print("=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
