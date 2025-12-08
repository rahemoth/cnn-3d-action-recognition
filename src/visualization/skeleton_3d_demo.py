"""3D skeleton visualization using matplotlib."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from src.visualization.pose_interface import (
    PoseSequence,
    SKELETON_CONNECTIONS,
    generate_mock_pose_sequence
)


class Skeleton3DVisualizer:
    """Visualize 3D skeleton poses using matplotlib."""
    
    def __init__(self, figsize=(10, 8)):
        """
        Args:
            figsize: Figure size (width, height)
        """
        self.figsize = figsize
        self.fig = None
        self.ax = None
    
    def plot_frame(
        self,
        pose_sequence: PoseSequence,
        frame_idx: int = 0,
        show: bool = True
    ):
        """
        Plot a single frame of the pose sequence.
        
        Args:
            pose_sequence: PoseSequence to visualize
            frame_idx: Frame index to plot
            show: Whether to display the plot
        """
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.clear()
        
        # Get keypoints for this frame
        keypoints = pose_sequence.get_frame(frame_idx)
        
        # Extract coordinates
        xs = [kp.x for kp in keypoints]
        ys = [kp.y for kp in keypoints]
        zs = [kp.z for kp in keypoints]
        
        # Plot keypoints
        self.ax.scatter(xs, ys, zs, c='red', marker='o', s=50)
        
        # Plot skeleton connections
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                x_line = [keypoints[start_idx].x, keypoints[end_idx].x]
                y_line = [keypoints[start_idx].y, keypoints[end_idx].y]
                z_line = [keypoints[start_idx].z, keypoints[end_idx].z]
                self.ax.plot(x_line, y_line, z_line, 'b-', linewidth=2)
        
        # Set labels and limits
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # Set consistent axis limits
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([0, 2])
        self.ax.set_zlim([-1, 1])
        
        # Set title
        title = f"Frame {frame_idx + 1}/{len(pose_sequence)}"
        if pose_sequence.action_label:
            title += f" - Action: {pose_sequence.action_label}"
        self.ax.set_title(title)
        
        if show:
            plt.show()
    
    def animate(
        self,
        pose_sequence: PoseSequence,
        interval: int = 33,
        save_path: str = None
    ):
        """
        Create an animated visualization of the pose sequence.
        
        Args:
            pose_sequence: PoseSequence to animate
            interval: Delay between frames in milliseconds
            save_path: Optional path to save animation (e.g., 'animation.gif')
        """
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        def update(frame_idx):
            """Update function for animation."""
            self.ax.clear()
            
            # Get keypoints for this frame
            keypoints = pose_sequence.get_frame(frame_idx)
            
            # Extract coordinates
            xs = [kp.x for kp in keypoints]
            ys = [kp.y for kp in keypoints]
            zs = [kp.z for kp in keypoints]
            
            # Plot keypoints
            self.ax.scatter(xs, ys, zs, c='red', marker='o', s=50, alpha=0.8)
            
            # Plot skeleton connections
            for start_idx, end_idx in SKELETON_CONNECTIONS:
                if start_idx < len(keypoints) and end_idx < len(keypoints):
                    x_line = [keypoints[start_idx].x, keypoints[end_idx].x]
                    y_line = [keypoints[start_idx].y, keypoints[end_idx].y]
                    z_line = [keypoints[start_idx].z, keypoints[end_idx].z]
                    self.ax.plot(x_line, y_line, z_line, 'b-', linewidth=2, alpha=0.6)
            
            # Set labels and limits
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_xlim([-1, 1])
            self.ax.set_ylim([0, 2])
            self.ax.set_zlim([-1, 1])
            
            # Set title
            title = f"Frame {frame_idx + 1}/{len(pose_sequence)}"
            if pose_sequence.action_label:
                title += f" - Action: {pose_sequence.action_label}"
            self.ax.set_title(title)
            
            return self.ax,
        
        # Create animation
        anim = FuncAnimation(
            self.fig,
            update,
            frames=len(pose_sequence),
            interval=interval,
            blit=False,
            repeat=True
        )
        
        if save_path:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=int(1000/interval))
            print("Animation saved!")
        
        plt.show()
        
        return anim


def demo_skeleton_visualization():
    """Run a demo of the 3D skeleton visualization."""
    print("3D Skeleton Visualization Demo")
    print("=" * 50)
    
    # Generate mock pose sequences for different actions
    actions = ["walking", "running", "jumping"]
    
    for action in actions:
        print(f"\nGenerating {action} pose sequence...")
        pose_seq = generate_mock_pose_sequence(
            action_label=action,
            num_frames=30,
            fps=30.0
        )
        
        print(f"Visualizing {action}...")
        visualizer = Skeleton3DVisualizer()
        
        # Plot first frame
        print(f"Showing first frame of {action}...")
        visualizer.plot_frame(pose_seq, frame_idx=0, show=False)
        
        # Animate the sequence
        print(f"Animating {action} (close window to continue)...")
        visualizer.animate(pose_seq, interval=33)
        
        plt.close('all')
    
    print("\nDemo completed!")


if __name__ == "__main__":
    demo_skeleton_visualization()
