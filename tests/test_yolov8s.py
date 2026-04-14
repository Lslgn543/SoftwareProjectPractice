"""
YOLOv8s 通用目标检测模型测试
"""

import cv2
import numpy as np
import os
import warnings
from pathlib import Path

# 抑制所有警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

class YOLOv8sDetector:
    """YOLOv8s 通用目标检测模型"""
    
    def __init__(self, model_path):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            # COCO 数据集类别标签
            self.class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
        except Exception:
            self.model = None
    
    def detect(self, image, conf_threshold=0.5):
        """执行目标检测"""
        if self.model is None:
            return []
        
        # 执行检测
        results = self.model(image, conf=conf_threshold)
        
        # 处理检测结果
        detections = []
        for result in results:
            for box in result.boxes:
                # 提取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 提取置信度
                confidence = float(box.conf[0])
                # 提取类别索引
                class_id = int(box.cls[0])
                # 获取类别名称
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'Class {class_id}'
                
                detections.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        return detections
    
    def __call__(self, image, conf_threshold=0.5):
        """执行目标检测（默认调用）"""
        return self.detect(image, conf_threshold)
    
    def close(self):
        """关闭模型"""
        if self.model:
            del self.model

def test_with_camera():
    """使用摄像头进行实时目标检测"""
    # 初始化模型
    print("正在初始化模型...")
    model_dir = Path(__file__).parent.parent
    
    yolo_detector = YOLOv8sDetector(
        str(model_dir / 'weights' / 'yolo' / 'yolov8s.pt')
    )
    
    if not yolo_detector.model:
        print("模型加载失败，无法继续测试")
        return
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("\n按 'q' 键退出")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        
        # 执行目标检测
        detections = yolo_detector.detect(frame, conf_threshold=0.4)
        
        # 绘制检测结果
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签和置信度
            label = f'{class_name}: {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示信息
        cv2.putText(frame, f'Detections: {len(detections)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, 'Press q to quit', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示结果
        cv2.imshow('YOLOv8s Object Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    yolo_detector.close()
    print("检测已停止")

def test_with_image(image_path):
    """使用图像进行目标检测测试"""
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        return
    
    # 初始化模型
    print("正在初始化模型...")
    model_dir = Path(__file__).parent.parent
    
    yolo_detector = YOLOv8sDetector(
        str(model_dir / 'weights' / 'yolo' / 'yolov8s.pt')
    )
    
    if not yolo_detector.model:
        print("模型加载失败，无法继续测试")
        return
    
    # 执行目标检测
    detections = yolo_detector.detect(image, conf_threshold=0.4)
    
    # 绘制检测结果
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        confidence = detection['confidence']
        class_name = detection['class_name']
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制标签和置信度
        label = f'{class_name}: {confidence:.2f}'
        cv2.putText(image, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 打印检测结果
    print(f"检测到 {len(detections)} 个目标")
    for i, detection in enumerate(detections):
        print(f"目标 {i+1}: {detection['class_name']} (置信度: {detection['confidence']:.2f})")
    
    # 显示结果
    cv2.imshow('YOLOv8s Object Detection - Image Test', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    yolo_detector.close()

if __name__ == '__main__':
    print("YOLOv8s 通用目标检测模型测试")
    print("=" * 50)
    
    # 测试选项
    print("1. 使用摄像头测试")
    print("2. 使用图像测试")
    choice = input("请选择测试方式 (1/2): ")
    
    if choice == '1':
        test_with_camera()
    elif choice == '2':
        image_path = input("请输入图像路径: ")
        test_with_image(image_path)
    else:
        print("无效选择")
