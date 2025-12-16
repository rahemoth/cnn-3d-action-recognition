# 使用示例 (Usage Examples)

## 基本使用 (Basic Usage)

### 1. 使用摄像头进行实时检测 (Real-time detection with webcam)

```python
# 运行主程序
python main.py

# 或者显式指定摄像头
python main.py --source 0
```

### 2. 处理视频文件 (Process video file)

```python
python main.py --source path/to/your/video.mp4
```

### 3. 运行演示程序 (Run demo)

```python
python demo.py
```

## 高级使用 (Advanced Usage)

### 自定义姿态检测器 (Custom Pose Detector)

```python
from pose_detector import PoseDetector

# 创建检测器，自定义置信度阈值
detector = PoseDetector(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 处理帧
results, rgb_frame = detector.detect(frame)
keypoints_3d = detector.extract_keypoints(results, frame.shape)
key_points = detector.get_key_body_points(keypoints_3d)

# 清理资源
detector.close()
```

### 自定义3D可视化 (Custom 3D Visualization)

```python
from skeleton_3d import Skeleton3D

# 创建3D骨架可视化器
skeleton = Skeleton3D(figsize=(12, 10))

# 更新显示
skeleton.update(key_points)

# 关闭
skeleton.close()
```

### 使用工具函数 (Using Utility Functions)

```python
from utils import (
    calculate_angle,
    calculate_distance,
    normalize_skeleton,
    get_body_center
)

# 计算关节角度（例如：肘部角度）
angle = calculate_angle(
    key_points['left_shoulder'],
    key_points['left_elbow'],
    key_points['left_wrist']
)
print(f"左肘角度: {angle}°")

# 计算身体中心
center = get_body_center(key_points)
print(f"身体中心: {center}")

# 归一化骨架
normalized = normalize_skeleton(key_points)
```

## 完整示例 (Complete Example)

### 处理单个图像 (Process a single image)

```python
import cv2
from pose_detector import PoseDetector
from skeleton_3d import Skeleton3D

# 读取图像
image = cv2.imread('image.jpg')

# 初始化
detector = PoseDetector()
skeleton = Skeleton3D()

# 检测姿态
results, _ = detector.detect(image)
annotated = detector.draw_landmarks(image, results)

# 提取3D关键点
keypoints_3d = detector.extract_keypoints(results, image.shape)
key_points = detector.get_key_body_points(keypoints_3d)

# 显示结果
cv2.imshow('2D Pose', annotated)
if key_points:
    skeleton.update(key_points)
    
cv2.waitKey(0)

# 清理
detector.close()
skeleton.close()
cv2.destroyAllWindows()
```

### 批量处理视频帧 (Batch process video frames)

```python
import cv2
from pose_detector import PoseDetector

cap = cv2.VideoCapture('video.mp4')
detector = PoseDetector()

# 存储所有帧的关键点
all_keypoints = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results, _ = detector.detect(frame)
    keypoints_3d = detector.extract_keypoints(results, frame.shape)
    
    if keypoints_3d is not None:
        all_keypoints.append(keypoints_3d)

cap.release()
detector.close()

print(f"处理了 {len(all_keypoints)} 帧")
```

## 性能优化建议 (Performance Optimization Tips)

1. **降低视频分辨率**
```python
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

2. **调整检测置信度**
```python
# 提高置信度减少误检测
detector = PoseDetector(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
```

3. **跳帧处理**
```python
frame_skip = 2  # 每隔2帧处理一次
frame_count = 0

while True:
    ret, frame = cap.read()
    frame_count += 1
    
    if frame_count % frame_skip != 0:
        continue
    
    # 处理帧
    results, _ = detector.detect(frame)
    # ...
```

## 故障排除 (Troubleshooting)

### 摄像头无法打开
```python
# 尝试不同的摄像头索引
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"摄像头 {i} 可用")
        cap.release()
```

### 检测效果不佳
- 确保光照充足
- 人物应完整出现在画面中
- 避免背景过于复杂
- 调整检测置信度阈值

### 3D可视化卡顿
- 降低matplotlib刷新频率
- 减小3D窗口尺寸
- 使用更快的后端（如Qt5Agg）

```python
import matplotlib
matplotlib.use('Qt5Agg')  # 使用Qt后端
```
