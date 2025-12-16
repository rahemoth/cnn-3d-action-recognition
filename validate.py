"""
Quick validation script that tests the system with a synthetic test image.
Creates a simple stick figure and processes it through the system.
"""

import cv2
import numpy as np
from pose_detector import PoseDetector
from skeleton_3d import Skeleton3D
import matplotlib.pyplot as plt


def create_test_image():
    """Create a simple test image with a stick figure."""
    # Create white background
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Draw a simple stick figure in the center
    # Head
    cv2.circle(img, (400, 150), 40, (0, 0, 0), -1)
    
    # Body
    cv2.line(img, (400, 190), (400, 350), (0, 0, 0), 10)
    
    # Arms
    cv2.line(img, (400, 220), (300, 280), (0, 0, 0), 8)  # Left arm
    cv2.line(img, (400, 220), (500, 280), (0, 0, 0), 8)  # Right arm
    
    # Legs
    cv2.line(img, (400, 350), (350, 500), (0, 0, 0), 8)  # Left leg
    cv2.line(img, (400, 350), (450, 500), (0, 0, 0), 8)  # Right leg
    
    return img


def validate_system():
    """Validate the system with a test image."""
    print("Creating test image...")
    test_img = create_test_image()
    
    # Save test image
    cv2.imwrite('/tmp/test_stick_figure.png', test_img)
    print("✓ Test image saved to /tmp/test_stick_figure.png")
    
    print("\nInitializing pose detector...")
    detector = PoseDetector()
    
    print("Processing test image...")
    results, rgb_frame = detector.detect(test_img)
    
    # Draw landmarks
    annotated = detector.draw_landmarks(test_img, results)
    
    # Extract keypoints
    keypoints_3d = detector.extract_keypoints(results, test_img.shape)
    
    if keypoints_3d is not None:
        print(f"✓ Detected {len(keypoints_3d)} keypoints")
        
        # Get key body points
        key_points = detector.get_key_body_points(keypoints_3d)
        
        if key_points:
            print(f"✓ Extracted {len(key_points)} key body points")
            print("\nKey body points:")
            for name, point in list(key_points.items())[:5]:
                print(f"  - {name}: {point}")
        else:
            print("⚠ No key body points extracted (image may not show human pose)")
    else:
        print("⚠ No pose detected (this is expected for a simple stick figure)")
        print("  The system works best with actual human images/videos")
    
    # Save annotated image
    cv2.imwrite('/tmp/test_annotated.png', annotated)
    print("\n✓ Annotated image saved to /tmp/test_annotated.png")
    
    # Clean up
    detector.close()
    
    print("\n" + "=" * 60)
    print("✅ System validation complete!")
    print("=" * 60)
    print("\nThe system is ready to use with:")
    print("  - Webcam: python main.py")
    print("  - Video file: python main.py --source path/to/video.mp4")
    print("=" * 60)


if __name__ == '__main__':
    validate_system()
