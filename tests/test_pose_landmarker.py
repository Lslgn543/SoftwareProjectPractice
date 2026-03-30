"""
MediaPipe Pose Landmarker 检测 - 摄像头测试
支持 MediaPipe 0.10+ 新版本 Tasks API
"""

import cv2
from pathlib import Path

# MediaPipe Tasks API (0.10+)
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


class PoseLandmarkerCamera:
    """使用 Tasks API 的摄像头姿态检测器"""
    def __init__(self):
        base_options = python.BaseOptions(
            model_asset_path=str(Path(__file__).parent.parent / 'weights' / 'mediapipe' / 'pose_landmarker_full.task')
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
    
    def detect(self, frame_bgr, timestamp_ms=0):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self.detector.detect_for_video(mp_image, int(timestamp_ms))


def draw_pose_landmarks(frame, result):
    """
    在帧上绘制姿态 landmark
    
    Args:
        frame: BGR 图像帧
        result: PoseLandmarker 检测结果
    """
    if not result.pose_landmarks:
        return frame
    
    h, w = frame.shape[:2]
    
    for pose in result.pose_landmarks:
        # 绘制关键点
        for landmark in pose:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        
        # 绘制骨架连接
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),  # 鼻子到耳朵
            (0, 4), (4, 5), (5, 6),          # 另一侧耳朵
            (11, 12),                         # 肩膀连接
            (11, 13), (13, 15), (15, 17), (15, 19),  # 左臂
            (12, 14), (14, 16), (16, 18), (16, 20),  # 右臂
            (11, 23), (12, 24),              # 肩膀到臀部
            (23, 24),                         # 臀部连接
            (23, 25), (25, 27), (27, 29), (27, 31),  # 左腿
            (24, 26), (26, 28), (28, 30), (28, 32),  # 右腿
        ]
        
        for idx1, idx2 in connections:
            if idx1 < len(pose) and idx2 < len(pose):
                pt1 = (int(pose[idx1].x * w), int(pose[idx1].y * h))
                pt2 = (int(pose[idx2].x * w), int(pose[idx2].y * h))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
    
    return frame


def test_with_camera():
    """
    使用摄像头进行实时姿态 landmark 检测
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("按 'q' 键退出")
    print("使用 MediaPipe Pose Landmarker (Tasks API)")
    
    detector = PoseLandmarkerCamera()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        result = detector.detect(frame, timestamp_ms)
        
        # 绘制检测结果
        frame = draw_pose_landmarks(frame, result)
        
        # 显示检测信息
        if result.pose_landmarks:
            num_poses = len(result.pose_landmarks)
            cv2.putText(frame, f"Poses: {num_poses}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('MediaPipe Pose Landmarker', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("检测已停止")


if __name__ == '__main__':
    print("MediaPipe Pose Landmarker 摄像头测试")
    print("=" * 50)
    test_with_camera()
