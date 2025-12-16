# 3D Body Motion Recognition

基于OpenCV和MediaPipe的实时3D人体动作识别与可视化系统。

## 功能特点

- ✅ 使用OpenCV和MediaPipe进行实时人体姿态检测
- ✅ 提取人体关键点的3D坐标（包括肢体端点）
- ✅ 实时3D骨架模型可视化
- ✅ 支持摄像头和视频文件输入
- ✅ 同时显示2D姿态检测和3D骨架模型

## 系统要求

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- Matplotlib

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/rahemoth/cnn-3d-action-recognition.git
cd cnn-3d-action-recognition
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 使用摄像头（默认）

```bash
python main.py
```

### 使用视频文件

```bash
python main.py --source path/to/video.mp4
```

### 运行演示

```bash
python demo.py
```

### 按键控制

- `q`: 退出程序

## 项目结构

```
cnn-3d-action-recognition/
├── main.py           # 主应用程序入口
├── pose_detector.py  # 姿态检测模块
├── skeleton_3d.py    # 3D骨架可视化模块
├── demo.py          # 演示示例
├── requirements.txt  # 项目依赖
└── README.md        # 项目文档
```

## 核心模块说明

### PoseDetector (pose_detector.py)

使用MediaPipe Pose进行人体姿态检测：
- 检测33个身体关键点
- 提取3D坐标（x, y, z）和可见度信息
- 提供关键肢体端点（头部、肩膀、肘部、手腕、臀部、膝盖、脚踝）

### Skeleton3D (skeleton_3d.py)

3D骨架可视化：
- 使用Matplotlib进行实时3D渲染
- 显示关键点和骨架连接
- 自动调整视图范围

### BodyMotion3D (main.py)

主应用程序：
- 整合姿态检测和3D可视化
- 处理视频输入（摄像头或文件）
- 提供实时预览

## 技术细节

### 姿态检测

使用MediaPipe Pose检测人体姿态，提供：
- 33个3D关键点
- 归一化坐标和深度信息
- 置信度评分

### 3D建模

从检测到的关键点构建3D骨架：
- 提取13个关键肢体端点
- 定义14条骨架连接
- 实时更新3D可视化

### 可视化

- 2D窗口：显示原始视频和检测到的姿态标注
- 3D窗口：显示交互式3D骨架模型
- 实时刷新，保持同步

## 示例输出

程序运行时会打开两个窗口：
1. **2D Pose Detection**: 显示原始视频和检测到的关键点
2. **3D Body Pose Visualization**: 显示3D骨架模型

## 性能优化

- 使用MediaPipe的GPU加速（如果可用）
- 优化的关键点提取
- 高效的3D渲染

## 故障排除

### 摄像头无法打开
- 确保摄像头已连接并且没有被其他程序占用
- 尝试更改摄像头索引：`python main.py --source 1`

### 依赖安装问题
- 确保Python版本为3.7或更高
- 使用虚拟环境：`python -m venv venv && source venv/bin/activate`

### 性能问题
- 降低视频分辨率
- 调整MediaPipe的检测置信度阈值

## 贡献

欢迎提交问题和拉取请求！

## 许可证

MIT License

## 致谢

- [MediaPipe](https://google.github.io/mediapipe/) - Google的机器学习解决方案
- [OpenCV](https://opencv.org/) - 计算机视觉库