"""
使用 MediaPipe Holistic 进行简单的表情分类测试
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

class MediaPipeExpressionClassifier:
    """使用 MediaPipe Holistic 进行表情分类"""
    
    def __init__(self):
        self.detector = None
        self._load_model()
        # 表情标签（英文）
        self.emotion_labels = ['Neutral', 'Happy', 'Sad', 'Surprised', 'Angry']
    
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
    
    def classify_expression(self, image):
        """
        使用 MediaPipe Holistic 进行表情分类
        
        Args:
            image: 输入图像
        
        Returns:
            表情标签、置信度和人脸边界框
        """
        if not MEDIAPIPE_AVAILABLE or self.detector is None:
            return 'Unknown', 0.0, None
        
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
            return 'Unknown', 0.0, None
        
        # 提取面部关键点
        face_landmarks = []
        for landmark in result.face_landmarks:
            # 将归一化坐标转换为像素坐标
            x = int(landmark.x * image_resized.shape[1])
            y = int(landmark.y * image_resized.shape[0])
            face_landmarks.append((x, y))
        
        # 计算人脸边界框
        if face_landmarks:
            x_coords = [x for x, y in face_landmarks]
            y_coords = [y for x, y in face_landmarks]
            x1 = min(x_coords)
            y1 = min(y_coords)
            x2 = max(x_coords)
            y2 = max(y_coords)
            
            # 添加一些边距
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(image_resized.shape[1], x2 + margin)
            y2 = min(image_resized.shape[0], y2 + margin)
            
            # 转换回原始图像尺寸
            scale_x = image.shape[1] / fixed_size[0]
            scale_y = image.shape[0] / fixed_size[1]
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            face_box = (x1, y1, x2, y2)
        else:
            face_box = None
        
        # 计算表情特征
        emotion, confidence = self._calculate_expression(face_landmarks, image_resized.shape, result)
        
        return emotion, confidence, face_box
    
    def _calculate_expression(self, landmarks, image_shape, result):
        """
        基于面部关键点计算表情
        
        Args:
            landmarks: 面部关键点列表
            image_shape: 图像形状
            result: MediaPipe Holistic 检测结果
        
        Returns:
            表情标签和置信度
        """
        # 检查关键点数量是否足够
        if len(landmarks) < 468:  # MediaPipe 面部关键点有 468 个
            return 'Unknown', 0.0
        
        # 提取关键的面部关键点
        # 眼睛
        left_eye = landmarks[33]  # 左眼中心
        right_eye = landmarks[263]  # 右眼中心
        
        # 眉毛
        left_eyebrow = landmarks[105]  # 左眉毛中心
        right_eyebrow = landmarks[334]  # 右眉毛中心
        
        # 嘴巴
        mouth_left = landmarks[61]  # 左嘴角
        mouth_right = landmarks[291]  # 右嘴角
        mouth_top = landmarks[13]  # 上嘴唇
        mouth_bottom = landmarks[14]  # 下嘴唇
        
        # 计算眼睛开合度
        eye_height = abs(left_eye[1] - right_eye[1])
        eye_width = abs(left_eye[0] - right_eye[0])
        eye_ratio = eye_height / eye_width if eye_width > 0 else 0
        
        # 计算眉毛高度
        left_eyebrow_height = left_eyebrow[1] - left_eye[1]
        right_eyebrow_height = right_eyebrow[1] - right_eye[1]
        eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2
        
        # 计算嘴巴开合度
        mouth_height = abs(mouth_bottom[1] - mouth_top[1])
        mouth_width = abs(mouth_right[0] - mouth_left[0])
        mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
        
        # 利用 MediaPipe Holistic 的面部表情分析
        # 注意：MediaPipe Holistic 本身不直接提供表情分析，但我们可以基于关键点进行更精细的分析
        
        # 计算微表情特征
        # 1. 眼睛区域的微表情
        left_eye_landmarks = landmarks[36:42]  # 左眼周围的关键点
        right_eye_landmarks = landmarks[42:48]  # 右眼周围的关键点
        
        # 计算眼睛周围的紧张度
        left_eye_area = self._calculate_area(left_eye_landmarks)
        right_eye_area = self._calculate_area(right_eye_landmarks)
        eye_tension = (left_eye_area + right_eye_area) / 2
        
        # 2. 嘴巴区域的微表情
        mouth_landmarks = landmarks[48:68]  # 嘴巴周围的关键点
        mouth_area = self._calculate_area(mouth_landmarks)
        
        # 3. 眉毛区域的微表情
        left_eyebrow_landmarks = landmarks[17:22]  # 左眉毛关键点
        right_eyebrow_landmarks = landmarks[22:27]  # 右眉毛关键点
        left_eyebrow_angle = self._calculate_angle(left_eyebrow_landmarks)
        right_eyebrow_angle = self._calculate_angle(right_eyebrow_landmarks)
        eyebrow_angle = (left_eyebrow_angle + right_eyebrow_angle) / 2
        
        # 综合分析表情
        if mouth_ratio > 0.3:  # 嘴巴张开较大
            if eye_ratio > 0.3:  # 眼睛也张开较大
                return 'Surprised', 0.8
            else:
                return 'Happy', 0.8
        elif mouth_ratio < 0.1:  # 嘴巴紧闭
            if eyebrow_height < -10:  # 眉毛下垂
                return 'Sad', 0.7
            elif eyebrow_height > 10 or eyebrow_angle > 30:  # 眉毛上扬或角度较大
                return 'Angry', 0.7
            else:
                return 'Neutral', 0.9
        else:
            # 分析微表情
            if eye_tension > 1000:  # 眼睛紧张，可能表示惊讶或恐惧
                return 'Surprised', 0.6
            elif mouth_area > 2000:  # 嘴巴区域较大，可能表示开心
                return 'Happy', 0.6
            else:
                return 'Neutral', 0.7
    
    def _calculate_area(self, landmarks):
        """
        计算多边形区域
        
        Args:
            landmarks: 多边形的顶点列表
        
        Returns:
            区域面积
        """
        if len(landmarks) < 3:
            return 0
        
        # 使用 shoelace 公式计算面积
        area = 0
        n = len(landmarks)
        for i in range(n):
            x1, y1 = landmarks[i]
            x2, y2 = landmarks[(i + 1) % n]
            area += (x1 * y2) - (x2 * y1)
        return abs(area) / 2
    
    def _calculate_angle(self, landmarks):
        """
        计算眉毛的角度
        
        Args:
            landmarks: 眉毛的关键点列表
        
        Returns:
            角度（度）
        """
        if len(landmarks) < 2:
            return 0
        
        # 计算眉毛的起始点和结束点
        start = landmarks[0]
        end = landmarks[-1]
        
        # 计算角度
        angle = np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0]))
        return angle
    
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
        
        return result_image
    
    def close(self):
        """关闭模型"""
        if self.detector:
            self.detector.close()

def test_with_camera():
    """使用摄像头进行实时表情分类"""
    # 初始化分类器
    print("正在初始化 MediaPipe 表情分类器...")
    classifier = MediaPipeExpressionClassifier()
    
    # 检查模型是否加载成功
    if not classifier.detector:
        print("错误：无法加载 MediaPipe Holistic 模型")
        return
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("\n按 'q' 键退出")
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break
        
        # 调整图像大小以提高处理速度
        frame_resized = cv2.resize(frame, (640, 480))
        
        # 进行表情分类
        emotion, confidence, face_box = classifier.classify_expression(frame_resized)
        
        # 显示人脸框选
        if face_box:
            x1, y1, x2, y2 = face_box
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 显示结果
        cv2.putText(frame_resized, f'Expression: {emotion} ({confidence:.2f})', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示信息
        cv2.putText(frame_resized, 'Press q to quit', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示结果
        cv2.imshow('MediaPipe Expression Classification', frame_resized)
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    classifier.close()
    print("检测已停止")

def test_with_image(image_path):
    """使用图像进行表情分类测试"""
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        return
    
    # 初始化分类器
    print("正在初始化 MediaPipe 表情分类器...")
    classifier = MediaPipeExpressionClassifier()
    
    # 检查模型是否加载成功
    if not classifier.detector:
        print("错误：无法加载 MediaPipe Holistic 模型")
        return
    
    # 进行表情分类
    emotion, confidence, face_box = classifier.classify_expression(image)
    
    # 打印结果
    print(f"Expression classification result: {emotion}, Confidence: {confidence:.2f}")
    
    # 绘制结果
    display_image = image.copy()
    
    # 显示人脸框选
    if face_box:
        x1, y1, x2, y2 = face_box
        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 显示表情结果
    cv2.putText(display_image, f'Expression: {emotion} ({confidence:.2f})', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 显示结果
    cv2.imshow('MediaPipe Expression Classification - Image Test', display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    classifier.close()

if __name__ == "__main__":
    print("MediaPipe Expression Classification Test")
    print("=" * 50)
    
    # Test options
    print("1. Test with camera")
    print("2. Test with image")
    choice = input("Please select test method (1/2): ")
    
    if choice == '1':
        test_with_camera()
    elif choice == '2':
        image_path = input("Please enter image path: ")
        test_with_image(image_path)
    else:
        print("Invalid choice")
