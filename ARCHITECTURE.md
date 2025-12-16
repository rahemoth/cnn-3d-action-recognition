# System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                          │
│                                                                       │
│  ┌─────────────────────┐              ┌─────────────────────┐      │
│  │   2D Pose Window    │              │   3D Skeleton       │      │
│  │   (OpenCV Display)  │              │   (Matplotlib 3D)   │      │
│  │                     │              │                     │      │
│  │  • Shows original   │              │  • Shows 3D model   │      │
│  │    video frame      │              │  • Interactive view │      │
│  │  • Draws 2D pose    │              │  • Real-time update │      │
│  │    landmarks        │              │                     │      │
│  └─────────────────────┘              └─────────────────────┘      │
└───────────────┬─────────────────────────────────┬───────────────────┘
                │                                 │
                │                                 │
┌───────────────▼─────────────────────────────────▼───────────────────┐
│                     Application Layer                                │
│                      (main.py)                                       │
│                                                                       │
│  • Video input handling (webcam/file)                               │
│  • Frame processing loop                                             │
│  • Coordinate pose detection and visualization                      │
│  • User interaction management                                      │
└───────────────┬─────────────────────────────────┬───────────────────┘
                │                                 │
                │                                 │
┌───────────────▼─────────────────┐   ┌──────────▼──────────────────┐
│    Pose Detection Module        │   │   3D Visualization Module   │
│    (pose_detector.py)           │   │   (skeleton_3d.py)          │
│                                 │   │                             │
│  • MediaPipe Pose integration   │   │  • Matplotlib 3D plotting   │
│  • Keypoint extraction (33pts) │   │  • Skeleton rendering       │
│  • 3D coordinate calculation    │   │  • Coordinate transform     │
│  • Key body points selection    │   │  • View management          │
└───────────────┬─────────────────┘   └─────────────────────────────┘
                │
                │
┌───────────────▼─────────────────┐
│      MediaPipe Library          │
│                                 │
│  • TensorFlow Lite backend      │
│  • Pose estimation model        │
│  • GPU acceleration (optional)  │
└─────────────────────────────────┘


Data Flow:
──────────

1. Video Input → Main Application
   ↓
2. Frame → Pose Detector → MediaPipe
   ↓
3. MediaPipe → 33 Keypoints (x, y, z, visibility)
   ↓
4. Keypoints → Extract 13 Key Body Points
   ↓
5. Split:
   a) 2D Display: Frame + Landmarks → OpenCV Window
   b) 3D Display: Key Points → 3D Skeleton → Matplotlib Window


Key Components:
───────────────

┌─────────────────────────────────────────────────────────────┐
│ PoseDetector                                                 │
├─────────────────────────────────────────────────────────────┤
│ + detect(frame) → results                                   │
│ + extract_keypoints(results) → keypoints_3d[33,4]          │
│ + get_key_body_points(keypoints_3d) → key_points{13}       │
│ + draw_landmarks(frame, results) → annotated_frame         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Skeleton3D                                                   │
├─────────────────────────────────────────────────────────────┤
│ + update(key_points) → visualize 3D skeleton               │
│ + setup_plot() → configure axes and view                   │
│ + close() → cleanup resources                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ BodyMotion3D                                                 │
├─────────────────────────────────────────────────────────────┤
│ + process_frame(frame) → annotated_frame, key_points       │
│ + run() → main application loop                            │
│ + cleanup() → release resources                            │
└─────────────────────────────────────────────────────────────┘


Body Keypoints (13):
────────────────────

     0: nose
       
  11 ●   ● 12   (left_shoulder, right_shoulder)
     │   │
  13 ●   ● 14   (left_elbow, right_elbow)
     │   │
  15 ●   ● 16   (left_wrist, right_wrist)
     
  23 ●   ● 24   (left_hip, right_hip)
     │   │
  25 ●   ● 26   (left_knee, right_knee)
     │   │
  27 ●   ● 28   (left_ankle, right_ankle)


Skeleton Connections (14):
──────────────────────────

Torso:
  • left_shoulder ←→ right_shoulder
  • left_shoulder ←→ left_hip
  • right_shoulder ←→ right_hip
  • left_hip ←→ right_hip

Head:
  • nose ←→ left_shoulder
  • nose ←→ right_shoulder

Arms:
  • left_shoulder ←→ left_elbow ←→ left_wrist
  • right_shoulder ←→ right_elbow ←→ right_wrist

Legs:
  • left_hip ←→ left_knee ←→ left_ankle
  • right_hip ←→ right_knee ←→ right_ankle
```
