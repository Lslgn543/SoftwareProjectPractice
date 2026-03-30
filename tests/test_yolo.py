"""
YOLOv8 目标检测 - 摄像头测试
"""

import cv2
from ultralytics import YOLO
from pathlib import Path

YOLO_AVAILABLE = True

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
