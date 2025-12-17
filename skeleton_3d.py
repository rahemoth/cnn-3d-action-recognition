# python
import numpy as np
import matplotlib

# 尝试几个常用交互后端，找到可用的再导入 pyplot
for _backend in ('Qt5Agg', 'TkAgg', 'QtAgg'):
    try:
        matplotlib.use(_backend, force=True)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Skeleton3D:
    """3D skeleton visualization using matplotlib."""

    def __init__(self, figsize=(10, 8), backend_hint=None):
        """
        Initialize 3D skeleton visualizer.
        Args:
            figsize: Size of the matplotlib figure
            backend_hint: optional string to force a backend (e.g. 'Qt5Agg')
        """
        # 可选：外部强制指定后端
        if backend_hint:
            try:
                matplotlib.use(backend_hint, force=True)
            except Exception:
                pass

        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('nose', 'left_shoulder'),
            ('nose', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
        ]

        self.setup_plot()

        # 确保非阻塞显示（在支持的后端上会弹出独立窗口）
        try:
            plt.show(block=False)
        except Exception:
            pass

    def setup_plot(self):
        """Setup the 3D plot with proper axes and labels."""
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Body Pose Visualization')
        self.ax.view_init(elev=20, azim=45)
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

        # 保证顺序一致
        keys = list(key_points.keys())
        points = np.array([key_points[k] for k in keys])

        x = points[:, 0]
        y = -points[:, 1]
        z = -points[:, 2]

        self.ax.scatter(x, y, z, c='red', marker='o', s=50)

        for start_name, end_name in self.connections:
            if start_name in key_points and end_name in key_points:
                start = key_points[start_name]
                end = key_points[end_name]
                xs = [start[0], end[0]]
                ys = [-start[1], -end[1]]
                zs = [-start[2], -end[2]]
                self.ax.plot(xs, ys, zs, 'b-', linewidth=2)

        if len(points) > 0:
            center_x, center_y, center_z = np.mean(x), np.mean(y), np.mean(z)
            range_val = 300
            self.ax.set_xlim([center_x - range_val, center_x + range_val])
            self.ax.set_ylim([center_y - range_val, center_y + range_val])
            self.ax.set_zlim([center_z - range_val, center_z + range_val])

        # 刷新并让 GUI 事件循环处理（必要）
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception:
            pass

    def close(self):
        """Close the visualization window."""
        try:
            plt.close(self.fig)
        except Exception:
            pass