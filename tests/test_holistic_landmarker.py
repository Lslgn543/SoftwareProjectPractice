"""
MediaPipe Holistic Landmarker 检测 - 摄像头测试
支持 MediaPipe 0.10+ 新版本 Tasks API

Holistic Landmarker 可以同时检测：
- 人脸关键点 (Face Mesh)
- 手部关键点 (左右手)
- 姿态关键点 (Pose)
- 躯干关键点 (Torso)
"""

import cv2
from pathlib import Path

# MediaPipe Tasks API (0.10+)
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


class HolisticLandmarkerCamera:
    """使用 Tasks API 的摄像头 Holistic 检测器"""
    def __init__(self):
        base_options = python.BaseOptions(
            model_asset_path=str(Path(__file__).parent.parent / 'weights' / 'mediapipe' / 'holistic_landmarker.task')
        )
        options = vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.detector = vision.HolisticLandmarker.create_from_options(options)
    
    def detect(self, frame_bgr, timestamp_ms=0):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self.detector.detect_for_video(mp_image, int(timestamp_ms))


def draw_holistic_landmarks(frame, result):
    """
    在帧上绘制 holistic landmark (人脸、手部、姿态)
    
    Args:
        frame: BGR 图像帧
        result: HolisticLandmarker 检测结果
    """
    if not result.pose_landmarks and not result.face_landmarks and not result.left_hand_landmarks and not result.right_hand_landmarks:
        return frame
    
    h, w = frame.shape[:2]
    
    # 绘制姿态关键点 (红色)
    if result.pose_landmarks:
        pose = result.pose_landmarks
        # 绘制关键点
        for landmark in pose:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        
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
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
    
    # 绘制人脸关键点 (绿色)
    if result.face_landmarks:
        face = result.face_landmarks
        for landmark in face:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    
    # 绘制左手关键点 (蓝色)
    if result.left_hand_landmarks:
        hand = result.left_hand_landmarks
        for landmark in hand:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
        
        # 绘制手指连接
        finger_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
            (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
            (5, 9), (9, 13), (13, 17)  # 手掌连接
        ]
        
        for idx1, idx2 in finger_connections:
            if idx1 < len(hand) and idx2 < len(hand):
                pt1 = (int(hand[idx1].x * w), int(hand[idx1].y * h))
                pt2 = (int(hand[idx2].x * w), int(hand[idx2].y * h))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 1)
    
    # 绘制右手关键点 (青色)
    if result.right_hand_landmarks:
        hand = result.right_hand_landmarks
        for landmark in hand:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (255, 255, 0), -1)
        
        # 绘制手指连接
        finger_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
            (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
            (5, 9), (9, 13), (13, 17)  # 手掌连接
        ]
        
        for idx1, idx2 in finger_connections:
            if idx1 < len(hand) and idx2 < len(hand):
                pt1 = (int(hand[idx1].x * w), int(hand[idx1].y * h))
                pt2 = (int(hand[idx2].x * w), int(hand[idx2].y * h))
                cv2.line(frame, pt1, pt2, (255, 255, 0), 1)
    
    return frame


def test_with_camera():
    """
    使用摄像头进行实时 holistic landmark 检测
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("按 'q' 键退出")
    print("使用 MediaPipe Holistic Landmarker (Tasks API)")
    print("检测内容：人脸 + 手部 + 姿态")
    
    detector = HolisticLandmarkerCamera()
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        result = detector.detect(frame, timestamp_ms)
        
        # 绘制检测结果
        frame = draw_holistic_landmarks(frame, result)
        
        # 显示检测信息
        info_y = 30
        if result.pose_landmarks:
            cv2.putText(frame, f"Pose: DETECTED", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            info_y += 30
        
        if result.face_landmarks:
            cv2.putText(frame, f"Face: DETECTED", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            info_y += 30
        
        if result.left_hand_landmarks:
            cv2.putText(frame, f"Left Hand: DETECTED", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            info_y += 30
        
        if result.right_hand_landmarks:
            cv2.putText(frame, f"Right Hand: DETECTED", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            info_y += 30
        
        frame_count += 1
        cv2.putText(frame, f"Frame: {frame_count}", (10, info_y + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('MediaPipe Holistic Landmarker', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("检测已停止")


if __name__ == '__main__':
    print("MediaPipe Holistic Landmarker 摄像头测试")
    print("=" * 50)
    test_with_camera()
