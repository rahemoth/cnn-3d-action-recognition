"""
Main application for real-time 3D body pose detection and visualization.
Processes video input and displays both 2D pose detection and 3D skeleton model.
"""

import cv2
import argparse
import sys
from pose_detector import PoseDetector
from skeleton_3d import Skeleton3D


class BodyMotion3D:
    """Main application for 3D body motion recognition and visualization."""
    
    def __init__(self, source=0):
        """
        Initialize the application.
        
        Args:
            source: Video source (0 for webcam, or path to video file)
        """
        self.source = source
        self.pose_detector = PoseDetector()
        self.skeleton_3d = Skeleton3D()
        self.cap = None
        
    def process_frame(self, frame):
        """
        Process a single frame.
        
        Args:
            frame: Input frame from video
            
        Returns:
            annotated_frame: Frame with pose landmarks drawn
            key_points: 3D key body points
        """
        # Detect pose
        results, rgb_frame = self.pose_detector.detect(frame)
        
        # Draw landmarks on frame
        annotated_frame = self.pose_detector.draw_landmarks(frame, results)
        
        # Extract 3D keypoints
        keypoints_3d = self.pose_detector.extract_keypoints(results, frame.shape)
        
        # Get key body points for 3D visualization
        key_points = self.pose_detector.get_key_body_points(keypoints_3d)
        
        return annotated_frame, key_points
    
    def run(self):
        """Run the main application loop."""
        # Open video source
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            print(f"Error: Cannot open video source {self.source}")
            return
        
        print("Starting 3D Body Motion Recognition...")
        print("Press 'q' to quit")
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    # Check if source is a video file (not webcam)
                    if isinstance(self.source, str) and self.source not in ['/dev/video0', '/dev/video1']:
                        # If video file ended, loop it
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("Error: Cannot read frame")
                        break
                
                # Process frame
                annotated_frame, key_points = self.process_frame(frame)
                
                # Display 2D pose detection
                cv2.imshow('2D Pose Detection', annotated_frame)
                
                # Update 3D skeleton visualization
                if key_points is not None:
                    self.skeleton_3d.update(key_points)
                
                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources and close windows."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.pose_detector.close()
        self.skeleton_3d.close()
        print("Resources released.")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description='3D Body Motion Recognition using OpenCV and MediaPipe'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Video source: 0 for webcam, or path to video file (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Convert source to integer if it's a digit
    source = int(args.source) if args.source.isdigit() else args.source
    
    # Create and run application
    app = BodyMotion3D(source=source)
    app.run()


if __name__ == '__main__':
    main()
