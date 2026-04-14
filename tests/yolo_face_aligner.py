"""
基于 YOLOv8-face 模型的人脸对齐模块
"""

import cv2
import numpy as np
import os
import warnings
from pathlib import Path

# 抑制所有警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

class YOLOFaceAligner:
    """基于 YOLOv8-face 模型的人脸对齐器"""
    
    def __init__(self, model_path):
        self.model = None
        self.previous_matrix = None  # 存储上一帧的变换矩阵，用于平滑
        self.smooth_factor = 0.7  # 平滑因子，值越大越稳定
        self._load_model(model_path)
    
    def _load_model(self, model_path):
        """加载 YOLOv8-face 模型"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except Exception:
            pass
    
    def detect_face(self, image, conf_threshold=0.5):
        """
        检测人脸并返回关键点
        
        Args:
            image: 输入图像
            conf_threshold: 置信度阈值
        
        Returns:
            人脸边界框和关键点，格式为 (box, keypoints)，如果未检测到人脸返回 (None, None)
        """
        if self.model is None:
            return None, None
        
        # 执行检测
        results = self.model(image, conf=conf_threshold)
        
        # 处理检测结果
        for result in results:
            if len(result.boxes) > 0:
                # 获取第一个检测到的人脸
                box = result.boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 提取关键点
                keypoints = []
                if result.keypoints is not None:
                    # 获取第一个检测到的人脸的关键点
                    kp = result.keypoints[0]
                    # YOLOv8-face 通常返回 5 个关键点：左眼、右眼、鼻子、左嘴角、右嘴角
                    for i, point in enumerate(kp.xy[0]):
                        if i >= 5:  # 只需要前 5 个关键点
                            break
                        x, y = point
                        keypoints.append((int(x), int(y)))
                
                return (x1, y1, x2, y2), keypoints
        
        return None, None
    
    def align_face(self, image):
        """
        使用 YOLOv8-face 对人脸进行对齐
        
        Args:
            image: 输入图像
        
        Returns:
            对齐后的人脸图像或 None
        """
        # 检测人脸和关键点
        box, keypoints = self.detect_face(image)
        if box is None:
            return None
        
        # 确保我们有足够的关键点
        if len(keypoints) < 5:
            # 如果关键点不足，尝试使用边界框中心作为参考
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 估计关键点位置
            eye_distance = (x2 - x1) * 0.3
            
            keypoints = [
                (int(center_x - eye_distance), int(center_y - (y2 - y1) * 0.2)),  # 左眼
                (int(center_x + eye_distance), int(center_y - (y2 - y1) * 0.2)),  # 右眼
                (int(center_x), int(center_y)),  # 鼻子
                (int(center_x - eye_distance * 0.8), int(center_y + (y2 - y1) * 0.2)),  # 左嘴角
                (int(center_x + eye_distance * 0.8), int(center_y + (y2 - y1) * 0.2))   # 右嘴角
            ]
        
        # 对人脸进行对齐
        try:
            aligned_face = self._align_face(image, keypoints)
            return aligned_face
        except Exception:
            return None
    
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
        
        # 确保我们有足够的关键点
        if len(landmarks) < 5:
            return None
        
        # 使用前 5 个关键点
        src_landmarks = np.array(landmarks[:5], dtype=np.float32)
        
        # 计算变换矩阵 - 使用完整的仿射变换，但手动调整以保持比例
        # 首先计算旋转角度和缩放比例
        # 计算源图像中双眼的距离
        src_eye_distance = np.linalg.norm(src_landmarks[0] - src_landmarks[1])
        # 计算目标图像中双眼的距离
        dst_eye_distance = np.linalg.norm(desired_landmarks[0] - desired_landmarks[1])
        # 计算缩放比例
        scale = dst_eye_distance / src_eye_distance
        
        # 计算旋转角度
        src_eye_center = (src_landmarks[0] + src_landmarks[1]) / 2
        dst_eye_center = (desired_landmarks[0] + desired_landmarks[1]) / 2
        
        # 计算旋转角度（基于眼睛连线）
        src_eye_angle = np.arctan2(src_landmarks[1][1] - src_landmarks[0][1], src_landmarks[1][0] - src_landmarks[0][0])
        dst_eye_angle = np.arctan2(desired_landmarks[1][1] - desired_landmarks[0][1], desired_landmarks[1][0] - desired_landmarks[0][0])
        angle = dst_eye_angle - src_eye_angle
        
        # 构建变换矩阵
        # 旋转矩阵
        rotation_matrix = np.array([
            [np.cos(angle) * scale, -np.sin(angle) * scale],
            [np.sin(angle) * scale, np.cos(angle) * scale]
        ])
        
        # 平移向量
        translation = dst_eye_center - rotation_matrix.dot(src_eye_center)
        
        # 构建完整的仿射变换矩阵
        current_matrix = np.hstack([rotation_matrix, translation.reshape(2, 1)])
        
        # 平滑处理 - 调整平滑因子，使其更加合理
        if self.previous_matrix is not None:
            # 使用较小的平滑因子，允许更快地适应人脸姿态变化
            current_matrix = 0.3 * self.previous_matrix + 0.7 * current_matrix
        
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
        if self.model:
            del self.model

# 创建全局对齐器实例
global_yolo_aligner = None

def get_yolo_aligner():
    """
    获取全局 YOLO 人脸对齐器实例
    
    Returns:
        YOLOFaceAligner 实例
    """
    global global_yolo_aligner
    if global_yolo_aligner is None:
        model_path = Path(__file__).parent.parent / 'weights' / 'yolo' / 'yolov8-face.pt'
        global_yolo_aligner = YOLOFaceAligner(str(model_path))
    return global_yolo_aligner

def align_face_with_yolo(image):
    """
    使用 YOLOv8-face 对人脸进行对齐
    
    Args:
        image: 输入图像
    
    Returns:
        对齐后的人脸图像或 None
    """
    aligner = get_yolo_aligner()
    return aligner.align_face(image)

# 测试函数
def test_yolo_face_aligner():
    """测试 YOLO 人脸对齐器"""
    import cv2
    
    # 初始化对齐器
    aligner = get_yolo_aligner()
    
    if not aligner.model:
        print("无法加载 YOLOv8-face 模型")
        return
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("按 'q' 键退出")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测人脸
        box, keypoints = aligner.detect_face(frame)
        
        # 绘制人脸边界框和关键点
        if box:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制关键点
            for (x, y) in keypoints:
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            
            # 对齐人脸
            aligned_face = aligner.align_face(frame)
            if aligned_face is not None:
                # 在画面右侧显示对齐后的人脸
                h, w = frame.shape[:2]
                aligned_h, aligned_w = aligned_face.shape[:2]
                
                # 调整对齐后人脸大小以适应显示
                resize_factor = min(150 / aligned_w, 150 / aligned_h)
                resized_aligned = cv2.resize(aligned_face, (int(aligned_w * resize_factor), int(aligned_h * resize_factor)))
                
                if 10 + resized_aligned.shape[0] < h and w - resized_aligned.shape[1] - 10 > 0:
                    frame[10:10+resized_aligned.shape[0], w-resized_aligned.shape[1]-10:w-10] = resized_aligned
                    cv2.putText(frame, "Aligned Face", (w-resized_aligned.shape[1]-10, 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示信息
        cv2.putText(frame, 'Press q to quit', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示结果
        cv2.imshow('YOLO Face Aligner', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    aligner.close()

if __name__ == '__main__':
    test_yolo_face_aligner()
