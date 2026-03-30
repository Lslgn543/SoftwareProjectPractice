"""
MediaPipe 面部 Landmark 检测 - 摄像头测试
支持 MediaPipe 0.10+ 新版本和旧版本
"""

import cv2
from pathlib import Path

# 检测 MediaPipe 版本并选择合适的 API
try:
    # MediaPipe 0.10+ 新 API (Tasks API)
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import mediapipe as mp
    NEW_API = True
    
    class FaceLandmarkerCamera:
        """使用新 API 的摄像头面部检测器"""
        def __init__(self):
            base_options = python.BaseOptions(
                model_asset_path=str(Path(__file__).parent.parent / 'weights' / 'mediapipe' / 'face_landmarker.task')
            )
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
        
        def detect(self, frame_bgr, timestamp_ms=0):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            return self.detector.detect_for_video(mp_image, int(timestamp_ms))
    
except ImportError:
    # 旧版本 API (Solutions API)
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    NEW_API = False


def detect_face_landmarks(image_path, show_result=False):
    """
    检测图像中的面部 landmark
    
    Args:
        image_path: 图像路径
        show_result: 是否显示结果
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像：{image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if NEW_API:
        # 使用新 API
        base_options = python.BaseOptions(
            model_asset_path=str(Path(__file__).parent.parent / 'weights' / 'mediapipe' / 'face_landmarker.task')
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = detector.detect(mp_image)
        
        if show_result:
            print(f"检测到 {len(result.face_landmarks)} 张人脸")
            if result.face_landmarks:
                print(f"Landmark 点数：{len(result.face_landmarks[0])}")
        
        return result
    else:
        # 使用旧 API
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks is None:
                print("未检测到人脸")
                return None
            
            if show_result:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                
                cv2.imshow('MediaPipe 面部 Landmark', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                face_landmarks = results.multi_face_landmarks[0]
                print(f"\n检测到 {len(results.multi_face_landmarks)} 张人脸")
                print(f"Landmark 点数：{len(face_landmarks.landmark)}")
            
            return results


def test_with_camera():
    """
    使用摄像头进行实时面部 landmark 检测
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("按 'q' 键退出")
    
    if NEW_API:
        # 使用新 API
        print("使用 MediaPipe 新 API (Tasks)")
        detector = FaceLandmarkerCamera()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            result = detector.detect(frame, timestamp_ms)
            
            if result.face_landmarks:
                for face_landmarks in result.face_landmarks:
                    for landmark in face_landmarks:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            cv2.imshow('MediaPipe 实时面部检测', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        # 使用旧 API
        print("使用 MediaPipe 旧 API (Solutions)")
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头画面")
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)
                
                if results.multi_face_landmarks is not None:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                
                cv2.imshow('MediaPipe 实时面部检测', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    cap.release()
    cv2.destroyAllWindows()
    print("检测已停止")


if __name__ == '__main__':
    print("MediaPipe 面部 Landmark 摄像头测试")
    print(f"API 版本：{'新 (Tasks)' if NEW_API else '旧 (Solutions)'}")
    print("=" * 50)
    test_with_camera()
