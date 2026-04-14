"""
YOLO 人脸检测器
用于检测图像中的人脸区域，并使用 BlazeFace 进行人脸对齐
"""

import cv2
import numpy as np
from pathlib import Path
import os
import warnings

# 抑制所有警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

YOLO_AVAILABLE = False
model = None
blaze_detector = None

# 尝试导入 YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    # 加载 YOLO 模型
    model_path = Path(__file__).parent.parent / 'weights' / 'yolo' / 'yolov8n.pt'
    if model_path.exists():
        model = YOLO(str(model_path))
    else:
        YOLO_AVAILABLE = False
except Exception as e:
    YOLO_AVAILABLE = False

# 加载 BlazeFace 模型
try:
    from blaze_face_detector import BlazeFaceDetector
    blaze_model_path = Path(__file__).parent.parent / 'weights' / 'blaze.onnx'
    if blaze_model_path.exists():
        blaze_detector = BlazeFaceDetector(str(blaze_model_path))
except Exception as e:
    pass


def detect_faces(image, confidence_threshold=0.5):
    """
    检测图像中的人脸
    
    Args:
        image: 输入图像
        confidence_threshold: 置信度阈值
    
    Returns:
        人脸边界框列表，格式为 [x1, y1, x2, y2]
    """
    if YOLO_AVAILABLE and model is not None:
        # 使用 YOLO 检测人脸
        results = model(image, verbose=False)
        boxes = []
        
        # 提取人脸边界框（类别 0 通常是 'person'）
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # 检查是否为人脸或人物
            if model.names[cls] == 'person' and conf > confidence_threshold:
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 裁剪人物区域
                person_roi = image[y1:y2, x1:x2]
                
                # 使用 BlazeFace 在人物区域内检测人脸
                if blaze_detector is not None:
                    blaze_faces = blaze_detector.detect_faces(person_roi, confidence_threshold=0.5)
                    for (blaze_bbox, landmarks) in blaze_faces:
                        # 将 BlazeFace 检测的边界框转换为原始图像坐标
                        face_x1 = x1 + blaze_bbox[0]
                        face_y1 = y1 + blaze_bbox[1]
                        face_x2 = x1 + blaze_bbox[2]
                        face_y2 = y1 + blaze_bbox[3]
                        # 确保边界框在图像范围内
                        face_x1 = max(0, face_x1)
                        face_y1 = max(0, face_y1)
                        face_x2 = min(image.shape[1], face_x2)
                        face_y2 = min(image.shape[0], face_y2)
                        # 确保边界框有效
                        if face_x2 > face_x1 and face_y2 > face_y1:
                            boxes.append([face_x1, face_y1, face_x2, face_y2])
                else:
                    # 如果 BlazeFace 不可用，使用人物边界框的上半部分作为人脸区域
                    # 计算边界框中心和大小
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    # 假设人脸在人物边界框的上半部分
                    face_height = int((y2 - y1) * 0.4)
                    face_width = int(face_height * 0.8)
                    # 计算人脸边界框
                    face_x1 = max(0, center_x - face_width // 2)
                    face_y1 = max(0, y1)
                    face_x2 = min(image.shape[1], center_x + face_width // 2)
                    face_y2 = min(image.shape[0], y1 + face_height)
                    # 确保边界框有效
                    if face_x2 > face_x1 and face_y2 > face_y1:
                        boxes.append([face_x1, face_y1, face_x2, face_y2])
        
        return boxes
    else:
        # 回退到 OpenCV 的 Haar 级联分类器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        boxes = []
        for (x, y, w, h) in faces:
            boxes.append([x, y, x + w, y + h])
        return boxes
