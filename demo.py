"""
Example usage and demonstration of the 3D body motion recognition system.
"""

from pose_detector import PoseDetector
from skeleton_3d import Skeleton3D
import cv2
import numpy as np


def demo_with_webcam():
    """Demo using webcam input."""
    print("Demo: Using webcam for real-time pose detection")
    print("Press 'q' to quit")
    
    detector = PoseDetector()
    skeleton = Skeleton3D()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect pose
        results, _ = detector.detect(frame)
        
        # Draw on frame
        annotated = detector.draw_landmarks(frame, results)
        cv2.imshow('2D Pose', annotated)
        
        # Extract and visualize 3D
        keypoints_3d = detector.extract_keypoints(results, frame.shape)
        key_points = detector.get_key_body_points(keypoints_3d)
        
        if key_points:
            skeleton.update(key_points)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    skeleton.close()


def demo_with_video(video_path):
    """Demo using video file input."""
    print(f"Demo: Processing video file: {video_path}")
    print("Press 'q' to quit")
    
    detector = PoseDetector()
    skeleton = Skeleton3D()
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            continue
        
        # Detect pose
        results, _ = detector.detect(frame)
        
        # Draw on frame
        annotated = detector.draw_landmarks(frame, results)
        cv2.imshow('2D Pose', annotated)
        
        # Extract and visualize 3D
        keypoints_3d = detector.extract_keypoints(results, frame.shape)
        key_points = detector.get_key_body_points(keypoints_3d)
        
        if key_points:
            skeleton.update(key_points)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):  # Slower playback for video
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    skeleton.close()


if __name__ == '__main__':
    # Run webcam demo by default
    demo_with_webcam()
    
    # To use with a video file, uncomment and specify path:
    # demo_with_video('path/to/your/video.mp4')
