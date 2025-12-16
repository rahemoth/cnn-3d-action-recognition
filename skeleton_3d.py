"""
3D skeleton visualization module.
Creates and displays 3D models of human body from keypoints.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Skeleton3D:
    """3D skeleton visualization using matplotlib."""
    
    def __init__(self, figsize=(10, 8)):
        """
        Initialize 3D skeleton visualizer.
        
        Args:
            figsize: Size of the matplotlib figure
        """
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Define skeleton connections (body segments)
        self.connections = [
            # Torso
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            
            # Head
            ('nose', 'left_shoulder'),
            ('nose', 'right_shoulder'),
            
            # Left arm
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            
            # Right arm
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            
            # Left leg
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            
            # Right leg
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
        ]
        
        self.setup_plot()
        
    def setup_plot(self):
        """Setup the 3D plot with proper axes and labels."""
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Body Pose Visualization')
        
        # Set initial view angle
        self.ax.view_init(elev=20, azim=45)
        
        # Enable interactive mode
        plt.ion()
        
    def update(self, key_points):
        """
        Update the 3D skeleton with new keypoints.
        
        Args:
            key_points: Dictionary of key body endpoints with 3D coordinates
        """
        if key_points is None:
            return
        
        self.ax.clear()
        self.setup_plot()
        
        # Extract all points
        points = np.array([key_points[key] for key in key_points.keys()])
        
        # Invert Y and Z for better visualization (OpenCV uses different coordinate system)
        x = points[:, 0]
        y = -points[:, 1]  # Invert Y
        z = -points[:, 2]  # Invert Z
        
        # Plot keypoints
        self.ax.scatter(x, y, z, c='red', marker='o', s=50)
        
        # Draw skeleton connections
        for start_name, end_name in self.connections:
            if start_name in key_points and end_name in key_points:
                start = key_points[start_name]
                end = key_points[end_name]
                
                xs = [start[0], end[0]]
                ys = [-start[1], -end[1]]  # Invert Y
                zs = [-start[2], -end[2]]  # Invert Z
                
                self.ax.plot(xs, ys, zs, 'b-', linewidth=2)
        
        # Set consistent axis limits for stable visualization
        if len(points) > 0:
            center_x, center_y, center_z = np.mean(x), np.mean(y), np.mean(z)
            range_val = 300  # Adjust based on your scale
            
            self.ax.set_xlim([center_x - range_val, center_x + range_val])
            self.ax.set_ylim([center_y - range_val, center_y + range_val])
            self.ax.set_zlim([center_z - range_val, center_z + range_val])
        
        # Refresh the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def close(self):
        """Close the visualization window."""
        plt.close(self.fig)
