import cv2
import numpy as np
import os
import sys
import argparse

# 导入各功能模块
from basic_preprocessing import run_preprocessing_pipeline
from color_detection import detect_color_targets, create_color_comparison
from shape_number_recognition import run_shape_recognition
from parameter_tuner import ParameterTuner


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='OPENPV 主程序')
    parser.add_argument('--input', type=str, default=None, help='输入图像路径')
    parser.add_argument('--output', type=str, default='output', help='输出结果目录')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--tune', type=str, choices=['color', 'edge', 'morphology', 'threshold'], help='启用参数调优模式')
    return parser.parse_args()


def show_menu():
    """显示菜单"""
    print("\n" + "="*60)
    print("OPENPV 主程序")
    print("="*60)
    print("1. 图像基础预处理")
    print("2. 颜色阈值色块识别")
    print("3. 几何图形与数字识别")
    print("4. 交互调参工具")
    print("0. 退出")
    print("="*60)
    try:
        choice = input("请输入选择 (0-4): ")
        return choice
    except KeyboardInterrupt:
        print("\n用户中断操作")
        return '0'

def main():
    """主函数"""
    while True:
        choice = show_menu()
        
        if choice == '0':
            print("退出程序...")
            return 0
        
        # 解析命令行参数，获取默认值
        args = parse_arguments()
        output_dir = args.output
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            if choice == '1':
                # 1. 图像基础预处理
                print("\n正在执行图像基础预处理...")
                try:
                    # 询问用户是否指定图像路径
                    input_path = input("请输入图像路径 (回车使用默认路径): ").strip()
                    if not input_path:
                        input_path = None
                    run_preprocessing_pipeline(input_path, output_dir=output_dir)
                    print("\n预处理完成！")
                except KeyboardInterrupt:
                    print("\n用户中断操作")
                except Exception as e:
                    print(f"预处理错误: {str(e)}", file=sys.stderr)
                
            elif choice == '2':
                # 2. 颜色阈值色块识别
                print("\n正在执行颜色阈值色块识别...")
                try:
                    # 询问用户是否指定图像路径
                    input_path = input("请输入图像路径 (回车使用默认路径): ").strip()
                    if not input_path:
                        # 使用默认图像路径
                        import color_detection
                        image_path = "images/color_test.jpg"
                        img = cv2.imread(image_path)
                        if img is None:
                            raise ValueError("无法读取默认图像，请检查路径是否正确")
                        detection_results = detect_color_targets(img, output_dir=output_dir)
                        comparison_img = create_color_comparison(img, detection_results)
                        cv2.imwrite(os.path.join(output_dir, "color_comparison.jpg"), comparison_img)
                    else:
                        image = cv2.imread(input_path)
                        if image is None:
                            raise ValueError("无法读取图像，请检查路径是否正确")
                        color_results = detect_color_targets(image, output_dir=output_dir)
                        comparison_img = create_color_comparison(image, color_results)
                        cv2.imwrite(os.path.join(output_dir, "color_comparison.jpg"), comparison_img)
                        detection_results = color_results
                    
                    # 打印颜色检测结果
                    print("="*50)
                    print("颜色检测结果：")
                    for color, data in detection_results.items():
                        info_list = data["info"]
                        print(f"\n{color.upper()} 目标数量: {len(info_list)}")
                        for info in info_list:
                            print(f"  目标:{info['id']} 中心坐标:{info['center']} 面积:{info['area']:.2f}")
                    print("\n颜色检测完成！")
                except KeyboardInterrupt:
                    print("\n用户中断操作")
                except Exception as e:
                    print(f"颜色检测错误: {str(e)}", file=sys.stderr)
                
            elif choice == '3':
                # 3. 几何图形与数字识别
                print("\n正在执行几何图形与数字识别...")
                try:
                    # 询问用户是否指定图像路径
                    input_path = input("请输入图像路径 (回车使用默认路径): ").strip()
                    if not input_path:
                        input_path = None
                    run_shape_recognition(output_dir=output_dir, image_path=input_path)
                    print("\n几何图形与数字识别完成！")
                except KeyboardInterrupt:
                    print("\n用户中断操作")
                except Exception as e:
                    print(f"几何图形与数字识别错误: {str(e)}", file=sys.stderr)
                
            elif choice == '4':
                # 4. 交互调参工具
                print("\n交互调参工具")
                print("1. 颜色阈值调优")
                print("2. 边缘检测调优")
                print("3. 形态学操作调优")
                print("4. 阈值处理调优")
                try:
                    tune_choice = input("请选择调参模式 (1-4): ")
                    
                    # 确定调参模式
                    mode_map = {
                        '1': 'color',
                        '2': 'edge',
                        '3': 'morphology',
                        '4': 'threshold'
                    }
                    
                    if tune_choice in mode_map:
                        mode = mode_map[tune_choice]
                        # 询问用户是否指定图像路径
                        input_path = input("请输入图像路径 (回车使用默认路径): ").strip()
                        if not input_path:
                            input_path = "images/color_test.jpg"
                        
                        # 创建调优工具实例
                        tuner = ParameterTuner()
                        tuner.mode = mode
                        # 运行调优工具
                        tuner.run(input_path)
                    else:
                        print("无效的选择！")
                except KeyboardInterrupt:
                    print("\n用户中断操作")
                except Exception as e:
                    print(f"调参工具错误: {str(e)}", file=sys.stderr)
                
            else:
                print("无效的选择，请重新输入！")
                
        except Exception as e:
            print(f"错误: {str(e)}", file=sys.stderr)
        
        # 询问用户是否继续
        try:
            continue_choice = input("\n是否继续操作？ (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("退出程序...")
                return 0
        except KeyboardInterrupt:
            print("\n用户中断操作")
            return 0
    


if __name__ == "__main__":
    sys.exit(main())

