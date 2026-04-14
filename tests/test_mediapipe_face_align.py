"""
测试 MediaPipe 直接进行人脸对齐
"""

import cv2
import numpy as np
from pathlib import Path
import os
import warnings

# 抑制所有警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# 导入 MediaPipe Tasks API
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    MEDIAPIPE_AVAILABLE = False

class MediaPipeFaceAligner:
    """MediaPipe 人脸对齐器"""
    
    def __init__(self):
        self.detector = None
        self.previous_matrix = None  # 存储上一帧的变换矩阵，用于平滑
        self.smooth_factor = 0.7  # 平滑因子，值越大越稳定
        self._load_model()
    
    def _load_model(self):
        """加载 MediaPipe Holistic Landmarker 模型"""
        if MEDIAPIPE_AVAILABLE:
            model_path = Path(__file__).parent.parent / 'weights' / 'mediapipe' / 'holistic_landmarker.task'
            if model_path.exists():
                base_options = python.BaseOptions(
                    model_asset_path=str(model_path)
                )
                options = vision.HolisticLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                )
                self.detector = vision.HolisticLandmarker.create_from_options(options)
    
    def align_face(self, image):
        """
        使用 MediaPipe 直接对人脸进行对齐
        
        Args:
            image: 输入图像
        
        Returns:
            对齐后的人脸图像或 None
        """
        if not MEDIAPIPE_AVAILABLE or self.detector is None:
            return None
        
        # 调整图像为固定大小，避免 MediaPipe 状态冲突
        fixed_size = (256, 256)
        image_resized = cv2.resize(image, fixed_size)
        
        # 转换为 RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # 使用 MediaPipe Holistic 进行检测
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.detector.detect(mp_image)
        
        # 检查是否检测到面部关键点
        if not result.face_landmarks:
            return None
        
        # 提取面部关键点
        face_landmarks = []
        for landmark in result.face_landmarks:
            # 将归一化坐标转换为像素坐标
            x = int(landmark.x * image_resized.shape[1])
            y = int(landmark.y * image_resized.shape[0])
            face_landmarks.append((x, y))
        
        # 选择关键的面部关键点（左眼、右眼、鼻子、左嘴角、右嘴角）
        # MediaPipe 面部关键点索引：33 (左眼), 263 (右眼), 1 (鼻子), 61 (左嘴角), 291 (右嘴角)
        key_landmarks = []
        key_indices = [33, 263, 1, 61, 291]
        for idx in key_indices:
            if idx < len(face_landmarks):
                key_landmarks.append(face_landmarks[idx])
        
        if len(key_landmarks) < 5:
            return None
        
        # 对人脸进行对齐
        aligned_face = self._align_face(image_resized, key_landmarks)
        
        return aligned_face
    
    def _align_face(self, image, landmarks):
        """
        对人脸进行对齐
        
        Args:
            image: 输入图像
            landmarks: 面部关键点列表
        
        Returns:
            对齐后的人脸图像
        """
        # 定义目标关键点位置（标准人脸）
        desired_landmarks = np.array([
            [30.2946, 51.6963],  # 左眼
            [65.5318, 51.5014],  # 右眼
            [48.0252, 71.7366],  # 鼻子
            [33.5493, 92.3655],  # 左嘴角
            [62.7299, 92.2041]   # 右嘴角
        ], dtype=np.float32)
        
        # 计算变换矩阵
        src_landmarks = np.array(landmarks, dtype=np.float32)
        current_matrix = cv2.estimateAffinePartial2D(src_landmarks, desired_landmarks)[0]
        
        # 平滑处理
        if self.previous_matrix is not None:
            # 使用加权平均来平滑变换矩阵
            current_matrix = self.smooth_factor * self.previous_matrix + (1 - self.smooth_factor) * current_matrix
        
        # 保存当前矩阵
        self.previous_matrix = current_matrix
        
        # 应用变换
        aligned_face = cv2.warpAffine(
            image, 
            current_matrix, 
            (112, 112),  # 输出大小
            flags=cv2.INTER_LINEAR
        )
        
        return aligned_face
    
    def draw_landmarks(self, image, landmarks):
        """
        在图像上绘制面部关键点
        
        Args:
            image: 输入图像
            landmarks: 面部关键点列表
        
        Returns:
            绘制了关键点的图像
        """
        result_image = image.copy()
        
        # 绘制关键点
        for (x, y) in landmarks:
            cv2.circle(result_image, (x, y), 2, (0, 255, 0), -1)
        
        # 绘制关键连接线
        key_indices = [33, 263, 1, 61, 291]
        for i in range(len(key_indices)):
            if key_indices[i] < len(landmarks):
                for j in range(i + 1, len(key_indices)):
                    if key_indices[j] < len(landmarks):
                        cv2.line(
                            result_image,
                            landmarks[key_indices[i]],
                            landmarks[key_indices[j]],
                            (0, 255, 0),
                            1
                        )
        
        return result_image
    
    def close(self):
        """关闭模型"""
        if self.detector:
            self.detector.close()

def test_mediapipe_face_align():
    """
    测试 MediaPipe 直接进行人脸对齐
    """
    # 初始化对齐器
    aligner = MediaPipeFaceAligner()
    
    # 检查模型是否加载成功
    if not aligner.detector:
        print("错误：无法加载 MediaPipe Holistic 模型")
        return
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("MediaPipe 人脸对齐测试开始")
    print("按 'q' 键退出")
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break
        
        # 调整图像大小以提高处理速度
        frame_resized = cv2.resize(frame, (640, 480))
        
        # 对人脸进行对齐
        aligned_face = aligner.align_face(frame_resized)
        
        # 显示结果
        display_frame = frame_resized.copy()
        
        if aligned_face is not None:
            # 将对齐后的人脸显示在右上角
            h, w = display_frame.shape[:2]
            aligned_h, aligned_w = aligned_face.shape[:2]
            
            # 确保对齐后的人脸大小适合显示
            if aligned_w <= w // 4 and aligned_h <= h // 4:
                display_frame[10:10+aligned_h, w-aligned_w-10:w-10] = aligned_face
                
                # 添加标签
                cv2.putText(
                    display_frame,
                    "Aligned Face",
                    (w-aligned_w-10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
        
        # 显示结果
        cv2.imshow('MediaPipe 人脸对齐', display_frame)
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    aligner.close()

if __name__ == "__main__":
    test_mediapipe_face_align()
