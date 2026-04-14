"""
YOLOv8 目标检测 - 摄像头测试
"""

import cv2
from ultralytics import YOLO
from pathlib import Path

YOLO_AVAILABLE = True
model = None

try:
    model_path = Path(__file__).parent.parent / 'weights' / 'yolo' / 'yolov8n.pt'
    if not model_path.exists():
        print(f"警告：模型文件不存在：{model_path}")
        print("请确保 yolov8n.pt 文件在 weights/yolo/ 目录下")
        YOLO_AVAILABLE = False
        model = None
    else:
        model = YOLO(str(model_path))
except Exception as e:
    print(f"加载 YOLO 模型失败：{e}")
    YOLO_AVAILABLE = False
    model = None


def detect_objects(image_path, show_result=False):
    """
    检测图像中的目标
    
    Args:
        image_path: 图像路径
        show_result: 是否显示结果
    """
    if not YOLO_AVAILABLE or model is None:
        print("YOLO 模型不可用")
        return None
    
    results = model(image_path)
    
    if show_result:
        result_img = results[0].plot()
        cv2.imshow('YOLO Detection', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        boxes = results[0].boxes
        print(f"\n检测到 {len(boxes)} 个目标:")
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"  - 类别：{model.names[cls]}, 置信度：{conf:.2f}")
    
    return results


def detect_faces(image, confidence_threshold=0.5):
    """
    使用 YOLO 检测图像中的人脸
    
    Args:
        image: 输入图像
        confidence_threshold: 置信度阈值
    
    Returns:
        人脸边界框列表，格式为 [x1, y1, x2, y2]
    """
    if not YOLO_AVAILABLE or model is None:
        print("YOLO 模型不可用，使用 OpenCV  Haar 级联分类器")
        # 回退到 OpenCV 的 Haar 级联分类器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        boxes = []
        for (x, y, w, h) in faces:
            boxes.append([x, y, x + w, y + h])
        return boxes
    
    results = model(image, verbose=False)
    boxes = []
    
    # 提取人脸边界框（类别 0 通常是 'person'，但我们需要更精确的人脸检测）
    # 注意：YOLOv8n 可能没有专门的人脸类别，这里我们使用 'person' 类别作为近似
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        # 检查是否为人脸或人物
        if model.names[cls] == 'person' and conf > confidence_threshold:
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])
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


def test_with_camera():
    """
    使用摄像头进行实时目标检测
    """
    if not YOLO_AVAILABLE or model is None:
        print("YOLO 模型不可用")
        return
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("按 'q' 键退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        
        results = model(frame, verbose=False)
        result_img = results[0].plot()
        
        cv2.imshow('YOLOv8 实时检测', result_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("检测已停止")


if __name__ == '__main__':
    print("YOLOv8 摄像头测试")
    print("=" * 50)
    test_with_camera()
