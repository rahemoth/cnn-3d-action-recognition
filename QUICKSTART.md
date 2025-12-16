# Quick Start Guide

## 快速开始 (Quick Start)

### 1. 安装 (Installation)

```bash
# Clone the repository
git clone https://github.com/rahemoth/cnn-3d-action-recognition.git
cd cnn-3d-action-recognition

# Install dependencies
pip install -r requirements.txt
```

### 2. 运行 (Run)

#### 选项 A: 使用摄像头 (Option A: Use Webcam)

```bash
python main.py
```

这将打开两个窗口：
- **2D Pose Detection**: 显示摄像头画面和检测到的姿态
- **3D Body Pose Visualization**: 显示3D骨架模型

#### 选项 B: 使用视频文件 (Option B: Use Video File)

```bash
python main.py --source path/to/your/video.mp4
```

#### 选项 C: 运行演示 (Option C: Run Demo)

```bash
python demo.py
```

### 3. 控制 (Controls)

- 按 `q` 键退出程序
- 3D窗口可以用鼠标拖动旋转视角

### 4. 预期输出 (Expected Output)

运行程序后，您将看到：

**2D窗口** (OpenCV):
```
┌─────────────────────────────────────┐
│                                     │
│     [实时视频画面]                   │
│                                     │
│     ●  ← 关键点标记                 │
│    ╱│╲                              │
│   ╱ │ ╲                             │
│  ●  ●  ●                            │
│     │                               │
│    ╱│╲                              │
│   ● │ ●                             │
│     │                               │
│    ╱ ╲                              │
│   ●   ●                             │
│                                     │
└─────────────────────────────────────┘
2D Pose Detection
```

**3D窗口** (Matplotlib):
```
┌─────────────────────────────────────┐
│  Z                                  │
│  ↑                                  │
│  │    ●  头部                       │
│  │   ╱│╲                            │
│  │  ● │ ● 手臂                      │
│  │    │                             │
│  │   ╱│╲                            │
│  │  ● │ ● 腿部                      │
│  │    │                             │
│  │   ╱ ╲                            │
│  │  ●   ●                           │
│  └────────→ X                       │
│   ╱                                 │
│  Y                                  │
└─────────────────────────────────────┘
3D Body Pose Visualization
```

### 5. 测试 (Testing)

```bash
# 运行单元测试
python test_system.py

# 运行验证
python validate.py
```

预期输出:
```
============================================================
Running 3D Body Motion Recognition System Tests
============================================================

Testing PoseDetector module...
✓ PoseDetector tests passed
Testing Skeleton3D module...
✓ Skeleton3D tests passed
Testing utility functions...
✓ Utility function tests passed
Testing main module...
✓ Main module tests passed
Testing demo module...
✓ Demo module tests passed

============================================================
✅ All tests passed successfully!
============================================================
```

### 6. 故障排除 (Troubleshooting)

#### 摄像头问题

```bash
# 尝试不同的摄像头索引
python main.py --source 1
python main.py --source 2
```

#### 性能问题

如果运行缓慢，可以：
1. 降低视频分辨率
2. 关闭其他占用资源的程序
3. 确保使用GPU加速（MediaPipe会自动使用）

#### 依赖问题

```bash
# 重新安装依赖
pip uninstall opencv-python mediapipe numpy matplotlib
pip install -r requirements.txt
```

### 7. 进阶使用 (Advanced Usage)

参见：
- `USAGE.md` - 详细使用示例
- `ARCHITECTURE.md` - 系统架构说明
- `SUMMARY.md` - 项目概述

### 8. 示例代码 (Example Code)

处理单张图片：

```python
import cv2
from pose_detector import PoseDetector
from skeleton_3d import Skeleton3D

# 初始化
image = cv2.imread('image.jpg')
detector = PoseDetector()
skeleton = Skeleton3D()

# 检测
results, _ = detector.detect(image)
annotated = detector.draw_landmarks(image, results)

# 3D显示
keypoints_3d = detector.extract_keypoints(results, image.shape)
key_points = detector.get_key_body_points(keypoints_3d)

cv2.imshow('2D', annotated)
skeleton.update(key_points)
cv2.waitKey(0)

# 清理
detector.close()
skeleton.close()
```

### 9. 系统要求 (System Requirements)

**最低配置**:
- CPU: Intel i3 或同等性能
- RAM: 4GB
- Python: 3.7+
- OS: Windows 10, Ubuntu 18.04, macOS 10.14+

**推荐配置**:
- CPU: Intel i5 或更高
- RAM: 8GB
- GPU: NVIDIA GPU with CUDA support (可选)
- Python: 3.8+

### 10. 性能指标 (Performance Metrics)

在推荐配置下：
- **帧率**: 20-30 FPS
- **延迟**: < 100ms
- **准确度**: MediaPipe Pose 标准精度
- **CPU占用**: 30-50%
- **内存占用**: ~500MB

---

**需要帮助？** 查看完整文档或提交issue: https://github.com/rahemoth/cnn-3d-action-recognition/issues
