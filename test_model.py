"""
模型测试主运行器
用于统一管理和运行各种模型测试
"""

import sys
from pathlib import Path


def print_menu():
    """打印测试菜单"""
    print("\n" + "=" * 50)
    print("模型测试菜单")
    print("=" * 50)
    print("1. MediaPipe 面部 Landmark 检测（摄像头）")
    print("2. YOLO 目标检测（摄像头）")
    print("3. MediaPipe Pose Landmark 检测（摄像头）")
    print("4. Halpe136 + MobileNet56 联合测试（摄像头）")
    print("5. MediaPipe Holistic Landmark 检测（摄像头）")
    print("6. 面部表情识别（摄像头）")
    print("7. MiniFASNetV2 活体检测（摄像头）")
    print("8. MobileNetV2 人脸特征提取（摄像头）")
    print("0. 退出")
    print("=" * 50)


def run_mediapipe_face_camera():
    """运行 MediaPipe 面部 landmark 摄像头检测"""
    from tests import test_mediapipe_face
    
    print("启动摄像头检测...")
    print("按 'q' 键退出")
    test_mediapipe_face.test_with_camera()


def run_yolo_camera():
    """运行 YOLO 目标检测（摄像头）"""
    from tests import test_yolo
    
    if not test_yolo.YOLO_AVAILABLE:
        print("错误：ultralytics 库未安装")
        print("请先运行：pip install ultralytics")
        return
    
    print("启动摄像头检测...")
    print("按 'q' 键退出")
    test_yolo.test_with_camera()


def run_pose_landmarker_camera():
    """运行 MediaPipe Pose Landmark 检测（摄像头）"""
    from tests import test_pose_landmarker
    
    print("启动 Pose Landmark 检测...")
    print("按 'q' 键退出")
    test_pose_landmarker.test_with_camera()


def run_halpe136_mobilenet():
    """运行 Halpe136 + MobileNet56 联合测试"""
    from tests import test_halpe136_mobilenet
    
    print("启动 Halpe136 姿态估计 + MobileNet56 人脸增强")
    print("按 'q' 键退出")
    print("按 'f' 键切换人脸增强")
    test_halpe136_mobilenet.test_with_camera()


def run_holistic_landmarker():
    """运行 MediaPipe Holistic Landmark 检测（摄像头）"""
    from tests import test_holistic_landmarker
    
    print("启动 Holistic Landmark 检测（人脸 + 手部 + 姿态）")
    print("按 'q' 键退出")
    test_holistic_landmarker.test_with_camera()


def run_facial_expression():
    """运行面部表情识别（摄像头）"""
    from tests import test_facial_expression
    
    print("启动面部表情识别")
    print("按 'q' 键退出")
    test_facial_expression.test_with_camera()


def run_minifasnet():
    """运行 MiniFASNetV2 活体检测（摄像头）"""
    from tests import test_minifasnet
    
    print("启动 MiniFASNetV2 活体检测")
    print("按 'q' 键退出")
    test_minifasnet.test_with_camera()


def run_mobilenetv2():
    """运行 MobileNetV2 人脸特征提取（摄像头）"""
    from tests import test_mobilenetv2
    
    print("启动 MobileNetV2 人脸特征提取")
    print("按 'q' 键退出")
    test_mobilenetv2.test_face_feature_extraction()


def main():
    """主函数"""
    print("欢迎使用模型测试工具！")
    
    while True:
        print_menu()
        
        choice = input("请选择测试项目 (0-5): ").strip()
        
        if choice == '1':
            run_mediapipe_face_camera()
        elif choice == '2':
            run_yolo_camera()
        elif choice == '3':
            run_pose_landmarker_camera()
        elif choice == '4':
            run_halpe136_mobilenet()
        elif choice == '5':
            run_holistic_landmarker()
        elif choice == '6':
            run_facial_expression()
        elif choice == '7':
            run_minifasnet()
        elif choice == '8':
            run_mobilenetv2()
        elif choice == '0':
            print("再见！")
            sys.exit(0)
        else:
            print("无效的选择，请重新输入")
        
        input("\n按回车键继续...")


if __name__ == '__main__':
    main()
