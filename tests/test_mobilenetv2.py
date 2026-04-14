"""
MobileNetV2 模型测试
使用 mobilenetv2.onnx 模型
支持分类和人脸特征提取功能
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# 从 MediaPipe 人脸对齐模块导入 align_face 函数和 get_aligner 函数
from mediapipe_face_aligner import align_face, get_aligner

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


class MobileNetV2Model:
    """MobileNetV2 模型"""
    
    def __init__(self, model_path):
        print("正在加载 MobileNetV2 模型...")
        self.session = ort.InferenceSession(model_path)
        print("✓ MobileNetV2 模型加载成功")
        # ImageNet 1000 类标签
        self.labels = self._load_imagenet_labels()
    
    def _load_imagenet_labels(self):
        """加载 ImageNet 标签"""
        # 简化版 ImageNet 标签（前 100 类）
        labels = [
            "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
            "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
            "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
            "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]
        return labels
    
    def preprocess(self, image):
        """预处理输入图像"""
        # MobileNetV2 输入尺寸为 112x112
        img = cv2.resize(image, (112, 112))
        img = img.astype(np.float32) / 255.0
        # 应用 ImageNet 均值和标准差
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # 添加 batch 维度
        img = img.astype(np.float32)  # 确保数据类型为 float32
        return img
    
    def classify(self, image):
        """执行图像分类"""
        input_tensor = self.preprocess(image)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        
        # 获取预测结果
        predictions = outputs[0][0]
        top_indices = np.argsort(predictions)[-5:][::-1]  # 前 5 个预测
        
        results = []
        for idx in top_indices:
            if idx < len(self.labels):
                label = self.labels[idx]
            else:
                label = f"class_{idx}"
            confidence = predictions[idx]
            results.append((label, confidence))
        
        return results
    
    def extract_features(self, image):
        """提取图像特征"""
        input_tensor = self.preprocess(image)
        input_name = self.session.get_inputs()[0].name
        
        # 获取模型的所有输出层名称
        output_names = [output.name for output in self.session.get_outputs()]
        
        # 如果有多个输出，使用第二个输出作为特征（通常是倒数第二层）
        if len(output_names) > 1:
            outputs = self.session.run([output_names[1]], {input_name: input_tensor})
        else:
            # 如果只有一个输出，使用该输出作为特征
            outputs = self.session.run(None, {input_name: input_tensor})
        
        # 获取特征向量
        features = outputs[0][0]
        # 归一化特征向量
        features = features / np.linalg.norm(features)
        
        return features
    
    def __call__(self, image):
        """执行模型推理（默认分类）"""
        return self.classify(image)


def draw_classification_result(frame, results):
    """绘制分类结果"""
    for i, (label, confidence) in enumerate(results[:3]):  # 只显示前 3 个
        cv2.putText(frame, f'{label}: {confidence:.2f}', (10, 30 + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


def draw_face_features(frame, boxes, feature_sizes):
    """绘制人脸特征提取结果"""
    for i, (box, feature_size) in enumerate(zip(boxes, feature_sizes)):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Face {i+1}: {feature_size}-dim feature', (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def test_with_camera():
    """使用摄像头进行实时分类"""
    # 初始化模型
    print("正在初始化模型...")
    model_dir = Path(__file__).parent.parent
    
    mobilenet_model = MobileNetV2Model(
        str(model_dir / 'weights' / 'mobilenetv2.onnx')
    )
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("\n按 'q' 键退出")
    print("按 'f' 键切换到人脸特征提取模式")
    
    feature_extraction_mode = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        
        if feature_extraction_mode:
            # 人脸检测
            boxes = detect_faces(frame)
            feature_sizes = []
            
            # 绘制结果
            display_frame = frame.copy()
            
            # 对每个检测到的人脸提取特征
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
                        features = mobilenet_model.extract_features(aligned_face)
                        feature_sizes.append(len(features))
                        
                        # 显示对齐前后的人脸
                        h, w = display_frame.shape[:2]
                        roi_h, roi_w = face_roi.shape[:2]
                        aligned_h, aligned_w = aligned_face.shape[:2]
                        
                        # 调整原始人脸大小以适应显示
                        resize_factor = min(150 / roi_w, 150 / roi_h)
                        resized_face = cv2.resize(face_roi, (int(roi_w * resize_factor), int(roi_h * resize_factor)))
                        
                        # 调整对齐后人脸大小以适应显示
                        aligned_resize_factor = min(150 / aligned_w, 150 / aligned_h)
                        resized_aligned = cv2.resize(aligned_face, (int(aligned_w * aligned_resize_factor), int(aligned_h * aligned_resize_factor)))
                        
                        # 在画面右侧显示原始人脸和对齐后的人脸
                        y_offset = 10 + (len(feature_sizes) - 1) * 180
                        if y_offset + resized_face.shape[0] < h and w - resized_face.shape[1] - 10 > 0:
                            # 显示原始人脸
                            display_frame[y_offset:y_offset+resized_face.shape[0], w-resized_face.shape[1]-10:w-10] = resized_face
                            cv2.putText(display_frame, "Original", (w-resized_face.shape[1]-10, y_offset-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                            # 显示对齐后的人脸
                            display_frame[y_offset+resized_face.shape[0]+10:y_offset+resized_face.shape[0]+10+resized_aligned.shape[0], w-resized_aligned.shape[1]-10:w-10] = resized_aligned
                            cv2.putText(display_frame, "Aligned", (w-resized_aligned.shape[1]-10, y_offset+resized_face.shape[0]),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        # 如果对齐失败，使用原始人脸区域
                        features = mobilenet_model.extract_features(face_roi)
                        feature_sizes.append(len(features))
            
            # 绘制人脸检测结果
            display_frame = draw_face_features(display_frame, boxes, feature_sizes)
            
            # 显示信息
            cv2.putText(display_frame, f'Faces: {len(boxes)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f'Mode: {"Feature Extraction" if feature_extraction_mode else "Classification"}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, 'Press f to switch mode, q to quit', (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # 执行分类
            results = mobilenet_model(frame)
            
            # 绘制结果
            display_frame = frame.copy()
            display_frame = draw_classification_result(display_frame, results)
            
            # 显示信息
            cv2.putText(display_frame, f'Mode: {"Feature Extraction" if feature_extraction_mode else "Classification"}', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, 'Press f to switch mode, q to quit', (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('MobileNetV2 Test', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            feature_extraction_mode = not feature_extraction_mode
            print(f"切换到 {'人脸特征提取' if feature_extraction_mode else '分类'} 模式")
    
    cap.release()
    cv2.destroyAllWindows()
    print("检测已停止")


def test_with_image(image_path):
    """使用图像进行测试"""
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        return
    
    # 初始化模型
    print("正在初始化模型...")
    model_dir = Path(__file__).parent.parent
    
    mobilenet_model = MobileNetV2Model(
        str(model_dir / 'weights' / 'mobilenetv2.onnx')
    )
    
    # 测试选项
    print("\n测试选项:")
    print("1. 图像分类")
    print("2. 人脸特征提取")
    choice = input("请选择测试类型 (1/2): ")
    
    if choice == '1':
        # 执行分类
        results = mobilenet_model(image)
        
        # 打印结果
        print("分类结果:")
        for label, confidence in results:
            print(f"{label}: {confidence:.2f}")
        
        # 绘制结果
        display_image = image.copy()
        display_image = draw_classification_result(display_image, results)
        
        # 显示结果
        cv2.imshow('MobileNetV2 Classification - Image Test', display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif choice == '2':
        # 人脸检测
        boxes = detect_faces(image)
        
        # 对每个检测到的人脸提取特征
        for i, box in enumerate(boxes):
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
                    features = mobilenet_model.extract_features(aligned_face)
                    print(f"人脸 {i+1} 特征向量维度: {len(features)}")
                    print(f"特征向量前 10 个值: {features[:10]}")
                else:
                    # 如果对齐失败，使用原始人脸区域
                    features = mobilenet_model.extract_features(face_roi)
                    print(f"人脸 {i+1} 特征向量维度: {len(features)}")
                    print(f"特征向量前 10 个值: {features[:10]}")
        
        # 绘制结果
        display_image = image.copy()
        display_image = draw_face_features(display_image, boxes, [len(mobilenet_model.extract_features(image[y1:y2, x1:x2])) for (x1, y1, x2, y2) in boxes])
        
        # 显示结果
        cv2.imshow('MobileNetV2 Face Feature Extraction - Image Test', display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("无效选择")


def test_face_feature_extraction():
    """专门测试人脸特征提取功能"""
    # 初始化模型
    print("正在初始化模型...")
    model_dir = Path(__file__).parent.parent
    
    mobilenet_model = MobileNetV2Model(
        str(model_dir / 'weights' / 'mobilenetv2.onnx')
    )
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("\n人脸特征提取测试")
    print("按 'q' 键退出")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        
        # 人脸检测
        boxes = detect_faces(frame)
        feature_sizes = []
        
        # 对每个检测到的人脸提取特征
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
                    features = mobilenet_model.extract_features(aligned_face)
                    feature_sizes.append(len(features))
                else:
                    # 如果对齐失败，使用原始人脸区域
                    features = mobilenet_model.extract_features(face_roi)
                    feature_sizes.append(len(features))
        
        # 绘制结果
        display_frame = frame.copy()
        display_frame = draw_face_features(display_frame, boxes, feature_sizes)
        
        # 显示信息
        cv2.putText(display_frame, f'Faces: {len(boxes)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, 'Face Feature Extraction Mode', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, 'Press q to quit', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('MobileNetV2 Face Feature Extraction', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("检测已停止")


if __name__ == '__main__':
    print("MobileNetV2 模型测试")
    print("=" * 50)
    
    # 测试选项
    print("1. 使用摄像头测试（支持分类和特征提取）")
    print("2. 使用图像测试（支持分类和特征提取）")
    print("3. 专门测试人脸特征提取（摄像头）")
    choice = input("请选择测试方式 (1/3): ")
    
    if choice == '1':
        test_with_camera()
    elif choice == '2':
        image_path = input("请输入图像路径: ")
        test_with_image(image_path)
    elif choice == '3':
        test_face_feature_extraction()
    else:
        print("无效选择")
