"""
Halpe136 姿态估计 + MobileNet56 人脸增强联合测试
Halpe136: 检测 136 个身体关键点（包括面部）
MobileNet56: 人脸超分辨率增强

参考 help.txt 中的正确实现方式
"""

import cv2
import numpy as np
import torch
from pathlib import Path


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32)):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)
    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(point, theta):
    x = point[0] * np.cos(theta) - point[1] * np.sin(theta)
    y = point[0] * np.sin(theta) + point[1] * np.cos(theta)
    return np.array([x, y], dtype=np.float32)


def _box_to_center_scale(x, y, w, h, aspect_ratio=1.0):
    pixel_std = 1
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25
    return center, scale


def _center_scale_to_box(center, scale):
    pixel_std = 1.0
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    return [xmin, ymin, xmin + w, ymin + h]


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float()
    if img.max() > 1:
        img /= 255
    return img


def heatmap_to_coord(heatmap, bbox, hm_shape=(64, 48)):
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
    heatmap = _sigmoid(heatmap)
    C, H, W = heatmap.shape
    preds = np.zeros((C, 2), dtype=np.float32)
    maxvals = np.zeros((C, 1), dtype=np.float32)
    for i in range(C):
        heatmap_channel = heatmap[i]
        pos = heatmap_channel.argmax()
        col = pos % W
        row = pos // W
        preds[i, 0] = col
        preds[i, 1] = row
        maxvals[i, 0] = heatmap_channel[row, col]
    preds = preds.astype(np.float32)
    maxvals = maxvals.astype(np.float32)
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    preds[:, 0] = preds[:, 0] / hm_shape[1] * width + xmin
    preds[:, 1] = preds[:, 1] / hm_shape[0] * height + ymin
    return preds, maxvals.flatten()


def _get_face_boxes(keypoints):
    """从关键点提取人脸边界框"""
    # Halpe136 中 26-93 是面部关键点
    face_keypoints = keypoints[:, 26:94]
    face_outline_keypoints = face_keypoints[:, :27]
    x_min = torch.min(face_outline_keypoints[:, :, 0], dim=1).values
    y_min = torch.min(face_outline_keypoints[:, :, 1], dim=1).values
    x_max = torch.max(face_outline_keypoints[:, :, 0], dim=1).values
    y_max = torch.max(face_outline_keypoints[:, :, 1], dim=1).values
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)


def crop_image(org_img, bbox, scale, out_w, out_h, return_box=False):
    src_h, src_w, _ = np.shape(org_img)
    x, y, box_w, box_h = bbox
    center_x, center_y = box_w / 2 + x, box_h / 2 + y
    aspect_src = box_w / box_h
    aspect_target = out_w / out_h
    if aspect_src > aspect_target:
        box_h = box_w / aspect_target
    else:
        box_w = box_h * aspect_target
    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
    new_width = box_w * scale
    new_height = box_h * scale
    left_top_x = center_x - new_width / 2
    left_top_y = center_y - new_height / 2
    right_bottom_x = center_x + new_width / 2
    right_bottom_y = center_y + new_height / 2
    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0
    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0
    if right_bottom_x > src_w - 1:
        left_top_x -= right_bottom_x - src_w + 1
        right_bottom_x = src_w - 1
    if right_bottom_y > src_h - 1:
        left_top_y -= right_bottom_y - src_h + 1
        right_bottom_y = src_h - 1
    dst_img = org_img[int(left_top_y): int(right_bottom_y) + 1, int(left_top_x): int(right_bottom_x) + 1]
    dst_img = cv2.resize(dst_img, (out_w, out_h))
    if return_box:
        return dst_img, [int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)]
    return dst_img


class Halpe136WithFaceAlign:
    """Halpe136 姿态估计 + MobileNet56 人脸对齐增强"""
    
    def __init__(self, alphapose_weights, face_aligner_weights=None, device='cuda',
                 input_size=(256, 192), output_size=(64, 48)):
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.eval_joints = list(range(136))
        
        # 加载 Halpe136 模型
        print("正在加载 Halpe136 模型...")
        
        # 先在 CPU 上加载，避免 CUDA 依赖问题
        if device == 'cpu':
            print("  使用 CPU 模式加载...")
            self.pose_model = torch.jit.load(alphapose_weights, map_location=torch.device('cpu'))
        else:
            try:
                self.pose_model = torch.jit.load(alphapose_weights)
            except Exception as e:
                print(f"  CUDA 加载失败，切换到 CPU: {e}")
                self.pose_model = torch.jit.load(alphapose_weights, map_location=torch.device('cpu'))
                self.device = 'cpu'
        
        self.pose_model.to(self.device)
        self.pose_model.eval()
        
        # Warmup
        print("  执行模型 warmup...")
        _ = self.pose_model(torch.zeros(1, 3, 256, 192).to(self.device))
        print("✓ Halpe136 模型加载成功")
        
        # 加载人脸对齐模型
        self.face_aligner = None
        if face_aligner_weights is not None:
            print("正在加载 MobileNet56 人脸增强模型...")
            try:
                self.face_aligner = MobileNetSEFaceAligner(face_aligner_weights, self.device)
                print("✓ 人脸增强已启用")
            except Exception as e:
                print(f"  人脸增强模型加载失败：{e}")
                print("  将只使用 Halpe136 姿态估计")
    
    def _preprocess(self, frame, detections):
        inps = []
        cropped_boxes = []
        for i in range(detections.shape[0]):
            box = detections[i, :4]
            inp, cropped_box = self._test_transform(frame, box)
            inps.append(torch.FloatTensor(inp))
            cropped_boxes.append(torch.FloatTensor(cropped_box))
        return torch.stack(inps, dim=0).to(self.device), torch.stack(cropped_boxes, dim=0).to(self.device)
    
    def _test_transform(self, src, bbox):
        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin,
            aspect_ratio=float(self.input_size[1]) / self.input_size[0]
        )
        scale = scale * 1.0
        inp_h, inp_w = self.input_size
        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox_box = _center_scale_to_box(center, scale)
        img = im_to_torch(img)
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)
        return img, bbox_box
    
    def __call__(self, frame, detections):
        """
        执行姿态估计和人脸增强
        
        Args:
            frame: BGR 图像
            detections: YOLO 检测结果 (N, 4) 边界框
            
        Returns:
            keypoints: 136 个关键点坐标
            scores: 置信度分数
        """
        empty_tensor = torch.tensor([])
        if detections.shape[0] <= 0:
            return empty_tensor, empty_tensor
        
        # 1. 预处理
        inps, cropped_boxes = self._preprocess(frame, detections)
        
        # 2. Halpe136 姿态估计
        with torch.no_grad():
            hm = self.pose_model(inps).cpu().detach()
        
        pose_coords = []
        pose_scores = []
        for i in range(hm.shape[0]):
            bbox = cropped_boxes[i].tolist()
            pose_coord, pose_score = heatmap_to_coord(
                hm[i][self.eval_joints], bbox, hm_shape=self.output_size
            )
            pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
            pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
        
        preds_kps = torch.cat(pose_coords)
        preds_scores = torch.cat(pose_scores)
        
        # 3. 人脸增强（如果启用了人脸增强器）
        if self.face_aligner is not None and preds_kps.shape[0] > 0:
            try:
                face_bboxes = _get_face_boxes(preds_kps)
                face_bboxes[:, 2:] = face_bboxes[:, 2:] - face_bboxes[:, :2]
                
                # 裁剪人脸区域
                face_cropped_and_bboxes = []
                for bbox in face_bboxes:
                    bbox_list = bbox.tolist()
                    if bbox_list[2] > 0 and bbox_list[3] > 0:
                        cropped_face, new_bbox = crop_image(
                            frame, bbox_list, 1.1, 56, 56, return_box=True
                        )
                        face_cropped_and_bboxes.append((cropped_face, new_bbox))
                
                if len(face_cropped_and_bboxes) > 0:
                    face_bboxes_tensor = torch.tensor(
                        [bbox for _, bbox in face_cropped_and_bboxes], dtype=torch.float32
                    )
                    face_bboxes_tensor[:, 2:] = face_bboxes_tensor[:, 2:] - face_bboxes_tensor[:, :2]
                    
                    # 人脸增强
                    face_cropped = np.array(
                        [cropped_img for cropped_img, _ in face_cropped_and_bboxes],
                        dtype=np.uint8
                    )
                    face_landmarks = self.face_aligner.align(face_cropped)
                    
                    # 转换回原图坐标
                    face_landmarks *= face_bboxes_tensor[:, 2:].unsqueeze(1)
                    face_landmarks += face_bboxes_tensor[:, :2].unsqueeze(1)
                    
                    # 替换 Halpe136 的脸部关键点（索引 26-93）
                    preds_kps[:, 26:94] = face_landmarks[:, :]
            except Exception as e:
                pass
        
        return preds_kps, preds_scores


class MobileNetSEFaceAligner:
    """MobileNet56 人脸对齐增强器"""
    mean = np.asarray([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
    std = np.asarray([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
    
    def __init__(self, weights_path, device):
        self.device = device
        print(f"  加载 MobileNet56 到 {device}...")
        
        # 在 CPU 上加载，避免 CUDA 依赖
        if device == 'cpu':
            self.model = torch.jit.load(weights_path, map_location=torch.device('cpu'))
        else:
            try:
                self.model = torch.jit.load(weights_path)
            except Exception as e:
                print(f"    CUDA 加载失败，切换到 CPU: {e}")
                self.model = torch.jit.load(weights_path, map_location=torch.device('cpu'))
        
        self.model.to(self.device)
        _ = self.model(torch.zeros(1, 3, 56, 56).to(self.device))
        print("  ✓ MobileNet56 加载完成")
    
    def preprocess(self, faces):
        faces = (faces / 255 - self.mean) / self.std
        return torch.from_numpy(faces.transpose((0, 3, 1, 2)))
    
    def align(self, faces):
        result = self.model(self.preprocess(faces).to(device=self.device, dtype=torch.float32)).detach().cpu()
        return result.view(-1, 68, 2)


def draw_keypoints(frame, keypoints, scores, threshold=0.3):
    """绘制身体和人脸关键点"""
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    
    for person_idx, (kps, score) in enumerate(zip(keypoints, scores)):
        color = colors[person_idx % len(colors)]
        
        # 1. 绘制面部关键点（索引 26-93，共 68 个点）
        for i in range(26, 94):
            if i < len(kps) and score[i] > threshold:
                x, y = int(kps[i][0]), int(kps[i][1])
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        
        # 2. 绘制身体关键点（索引 0-25，共 26 个点）
        for i in range(26):
            if i < len(kps) and score[i] > threshold:
                x, y = int(kps[i][0]), int(kps[i][1])
                cv2.circle(frame, (x, y), 3, color, -1)
    
    return frame


def test_with_camera():
    """使用摄像头进行测试"""
    try:
        from ultralytics import YOLO
        YOLO_AVAILABLE = True
    except ImportError:
        print("错误：ultralytics 未安装")
        print("请先运行：pip install ultralytics")
        YOLO_AVAILABLE = False
    
    if not YOLO_AVAILABLE:
        return
    
    # 初始化模型
    print("正在初始化模型...")
    model_dir = Path(__file__).parent.parent
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备：{device}")
    
    pose_model = Halpe136WithFaceAlign(
        alphapose_weights=str(model_dir / 'weights' / 'halpe' / 'halpe136_mobile.torchscript.pth'),
        face_aligner_weights=str(model_dir / 'weights' / 'mobilenet' / 'mobilenet56_se_external_model_best.torchscript.pth'),
        device=device
    )
    
    yolo_model = YOLO(str(model_dir / 'weights' / 'yolo' / 'yolov8n.pt'))
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("\n按 'q' 键退出")
    print("按 'f' 键切换人脸增强")
    
    enable_face_align = True
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        
        # YOLOv8 检测人体
        results = yolo_model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        
        # 只保留人（类别 0）
        person_mask = classes == 0
        person_boxes = boxes[person_mask]
        person_confs = confs[person_mask]
        
        # Halpe136 姿态估计 + 人脸增强
        if len(person_boxes) > 0:
            keypoints, scores = pose_model(frame, person_boxes)
        else:
            keypoints = torch.tensor([])
            scores = torch.tensor([])
        
        # 可视化
        display_frame = frame.copy()
        
        # 绘制 YOLO 检测到的人体框
        for box, conf in zip(person_boxes, person_confs):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display_frame, f'Person {conf:.2f}', (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 绘制关键点
        if len(keypoints) > 0:
            display_frame = draw_keypoints(display_frame, keypoints.numpy(), scores.numpy())
        
        # 显示信息
        cv2.putText(display_frame, f'Persons: {len(keypoints)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f'Face Align: {"ON" if enable_face_align else "OFF"}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Halpe136 + Face Align', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            enable_face_align = not enable_face_align
            print(f"人脸增强：{'开启' if enable_face_align else '关闭'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("检测已停止")


if __name__ == '__main__':
    print("Halpe136 + MobileNet56 联合测试")
    print("=" * 50)
    test_with_camera()
