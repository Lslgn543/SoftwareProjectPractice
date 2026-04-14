"""
MediaPipe 人脸对齐模块
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
    
    def close(self):
        """关闭模型"""
        if self.detector:
            self.detector.close()

# 创建全局对齐器实例
global_aligner = None

def get_aligner():
    """
    获取全局 MediaPipe 人脸对齐器实例
    
    Returns:
        MediaPipeFaceAligner 实例
    """
    global global_aligner
    if global_aligner is None:
        global_aligner = MediaPipeFaceAligner()
    return global_aligner

def align_face(image):
    """
    使用 MediaPipe 对人脸进行对齐
    
    Args:
        image: 输入图像
    
    Returns:
        对齐后的人脸图像或 None
    """
    aligner = get_aligner()
    return aligner.align_face(image)
