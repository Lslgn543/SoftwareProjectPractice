"""
人脸识别模型测试
使用 w600k_mbf.onnx 模型
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import os
import warnings

# 抑制所有警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# 从 MediaPipe 人脸对齐模块导入 align_face 函数和 get_aligner 函数
from mediapipe_face_aligner import align_face, get_aligner

class FaceRecognitionModel:
    """人脸识别模型"""
    
    def __init__(self, model_path):
        print("正在加载人脸识别模型...")
        self.session = ort.InferenceSession(model_path)
        print("✓ 人脸识别模型加载成功")
    
    def preprocess(self, face_image):
        """预处理人脸图像"""
        # 模型输入尺寸通常为 112x112
        img = cv2.resize(face_image, (112, 112))
        img = img.astype(np.float32) / 255.0
        # 应用均值和标准差
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # 添加 batch 维度
        img = img.astype(np.float32)  # 确保数据类型为 float32
        return img
    
    def extract_features(self, face_image):
        """提取人脸特征"""
        input_tensor = self.preprocess(face_image)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        
        # 获取特征向量
        features = outputs[0][0]
        # 归一化特征向量
        features = features / np.linalg.norm(features)
        
        return features
    
    def __call__(self, face_image):
        """执行模型推理（默认提取特征）"""
        return self.extract_features(face_image)

# 直接使用 MediaPipe 进行人脸检测和对齐
def detect_faces(image):
    """使用 MediaPipe 进行人脸检测"""
    aligner = get_aligner()
    
    # 调整图像为固定大小，避免 MediaPipe 状态冲突
    fixed_size = (256, 256)
    image_resized = cv2.resize(image, fixed_size)
    
    # 转换为 RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    # 使用 MediaPipe Holistic 进行检测
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = aligner.detector.detect(mp_image)
        
        # 检查是否检测到面部关键点
        if result.face_landmarks:
            # 提取面部关键点并计算人脸边界框
            face_landmarks = []
            for landmark in result.face_landmarks:
                # 将归一化坐标转换为像素坐标
                x = int(landmark.x * image_resized.shape[1])
                y = int(landmark.y * image_resized.shape[0])
                face_landmarks.append((x, y))
            
            # 计算边界框
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
                
                return [(x1, y1, x2, y2)]
    except Exception:
        pass
    
    # 如果检测失败，返回整个图像
    return [(0, 0, image.shape[1], image.shape[0])]

def calculate_similarity(feature1, feature2):
    """计算两个特征向量的相似度"""
    return np.dot(feature1, feature2)

def test_with_camera():
    """使用摄像头进行实时人脸识别"""
    # 初始化模型
    print("正在初始化模型...")
    model_dir = Path(__file__).parent.parent
    
    face_recognition_model = FaceRecognitionModel(
        str(model_dir / 'weights' / 'w600k_mbf.onnx')
    )
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("\n按 'q' 键退出")
    print("按 's' 键保存当前人脸特征")
    
    # 存储已知人脸特征
    known_features = []
    known_names = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        
        # 1. 人脸检测
        boxes = detect_faces(frame)
        
        # 2. 对每个检测到的人脸提取特征
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # 确保边界框在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                # 裁剪人脸区域
                face_roi = frame[y1:y2, x1:x2]
                
                # 使用 MediaPipe 对人脸进行对齐
                aligned_face = align_face(face_roi)
                if aligned_face is not None:
                    # 使用对齐后的人脸提取特征
                    features = face_recognition_model.extract_features(aligned_face)
                    
                    # 查找最相似的已知人脸
                    if known_features:
                        similarities = [calculate_similarity(features, known_feature) for known_feature in known_features]
                        max_similarity = max(similarities)
                        max_index = similarities.index(max_similarity)
                        
                        if max_similarity > 0.6:
                            name = known_names[max_index]
                            confidence = max_similarity
                        else:
                            name = "Unknown"
                            confidence = 0.0
                    else:
                        name = "Unknown"
                        confidence = 0.0
                    
                    # 显示对齐前后的人脸
                    h, w = frame.shape[:2]
                    roi_h, roi_w = face_roi.shape[:2]
                    aligned_h, aligned_w = aligned_face.shape[:2]
                    
                    # 调整原始人脸大小以适应显示
                    resize_factor = min(120 / roi_w, 120 / roi_h)
                    resized_face = cv2.resize(face_roi, (int(roi_w * resize_factor), int(roi_h * resize_factor)))
                    
                    # 调整对齐后人脸大小以适应显示
                    aligned_resize_factor = min(120 / aligned_w, 120 / aligned_h)
                    resized_aligned = cv2.resize(aligned_face, (int(aligned_w * aligned_resize_factor), int(aligned_h * aligned_resize_factor)))
                    
                    # 在画面右侧显示原始人脸和对齐后的人脸
                    y_offset = 10
                    if y_offset + resized_face.shape[0] < h and w - resized_face.shape[1] - 10 > 0:
                        # 显示原始人脸
                        frame[y_offset:y_offset+resized_face.shape[0], w-resized_face.shape[1]-10:w-10] = resized_face
                        cv2.putText(frame, "Original", (w-resized_face.shape[1]-10, y_offset-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # 显示对齐后的人脸
                        frame[y_offset+resized_face.shape[0]+10:y_offset+resized_face.shape[0]+10+resized_aligned.shape[0], w-resized_aligned.shape[1]-10:w-10] = resized_aligned
                        cv2.putText(frame, "Aligned", (w-resized_aligned.shape[1]-10, y_offset+resized_face.shape[0]),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # 绘制结果
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{name} ({confidence:.2f})', (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # 如果对齐失败，显示失败信息
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Face alignment failed", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 显示信息
        cv2.putText(frame, f'Faces: {len(boxes)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, 'Press s to save face, q to quit', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示结果
        cv2.imshow('Face Recognition', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and boxes:
            # 保存当前人脸特征
            box = boxes[0]
            x1, y1, x2, y2 = box
            face_roi = frame[y1:y2, x1:x2]
            aligned_face = align_face(face_roi)
            if aligned_face is not None:
                features = face_recognition_model.extract_features(aligned_face)
                name = input("请输入人脸名称: ")
                known_features.append(features)
                known_names.append(name)
                print(f"已保存人脸: {name}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("检测已停止")

def test_with_image(image_path):
    """使用图像进行人脸识别测试"""
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        return
    
    # 初始化模型
    print("正在初始化模型...")
    model_dir = Path(__file__).parent.parent
    
    face_recognition_model = FaceRecognitionModel(
        str(model_dir / 'weights' / 'w600k_mbf.onnx')
    )
    
    # 1. 人脸检测
    boxes = detect_faces(image)
    
    # 2. 对每个检测到的人脸提取特征
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # 确保边界框在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        if x2 > x1 and y2 > y1:
            # 裁剪人脸区域
            face_roi = image[y1:y2, x1:x2]
            
            # 使用 MediaPipe 对人脸进行对齐
            aligned_face = align_face(face_roi)
            if aligned_face is not None:
                # 使用对齐后的人脸提取特征
                features = face_recognition_model.extract_features(aligned_face)
                print(f"人脸特征向量维度: {len(features)}")
                print(f"特征向量前 10 个值: {features[:10]}")
            else:
                print("人脸对齐失败")
            
            # 绘制结果
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 显示结果
    cv2.imshow('Face Recognition - Image Test', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("人脸识别模型测试")
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
