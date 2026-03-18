import numpy as np
import cv2
import os
import sys

def apply_histogram_equalization(image):
    if len(image.shape) == 2:
        # 灰度图直接均衡化
        equalized = cv2.equalizeHist(image)
    else:
        # 彩色图转换为YUV空间，仅对亮度通道均衡化
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    return equalized

def create_processing_comparison(original, processed_list, titles):
    # 统一转换为BGR格式以便拼接
    display_images = [original]
    for img in processed_list:
        if len(img.shape) == 2:
            # 灰度图转BGR
            img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_display = img
        display_images.append(img_display)  
    # 统一尺寸
    height, width = original.shape[:2]
    resized_images = []
    for img in display_images:
        resized = cv2.resize(img, (width, height))
        resized_images.append(resized)
    # 水平拼接
    comparison = np.hstack(resized_images)
    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_color = (255, 255, 255) # 白色文字
    bg_color = (0, 0, 0)         # 黑色背景 
    # 计算每个标题的位置
    section_width = width
    for i, title in enumerate(titles):
        # 获取文本宽度和高度，用于绘制背景
        (text_width, text_height), baseline = cv2.getTextSize(title, font, font_scale, thickness)
        # 居中
        x = int(i * section_width + (section_width - text_width) / 2)
        y = 30
        cv2.rectangle(comparison, (x - 5, y - text_height - baseline - 5), 
                      (x + text_width + 5, y + 5), bg_color, -1)
        cv2.putText(comparison, title, (x, y), font, font_scale, text_color, thickness)

    return comparison

def save_images(img_bgr, img_gray, img_blur, equalized_img, comparison_img, img_name, save_dir="output"):
    # 检查并创建目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 提取原始文件名
    base_name = os.path.splitext(img_name)[0]
    # 1. 保存原始图像
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_original.png"), img_bgr)
    # 2. 保存灰度图
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_grayscale.png"), img_gray)
    # 3. 保存高斯模糊图
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_gaussian_blur.png"), img_blur)
    # 4. 保存直方图均衡化图
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_equalized.png"), equalized_img) 
    # 5. 保存拼接后的对比图
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_comparison.png"), comparison_img)
    print(f"所有图像已保存到：{os.path.abspath(save_dir)}")

def convert_to_grayscale(img_bgr):
      # 3. 图像处理流程
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # 灰度转换
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)  # 高斯模糊去噪
    equalized_img = apply_histogram_equalization(img_gray)  # 直方图均衡化
    return img_gray,img_blur,equalized_img

def run_preprocessing_pipeline(image_path=None, output_dir="output"):
    # 1. 处理参数
    if image_path is None:
        # 使用默认测试图像
        image_path = "images/basic_test.jpg"
    
    # 2. 读取并校验图像 (cv2.imread 返回 BGR 格式)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"图像读取失败: {image_path}")
        return
    
    img_gray, img_blur, equalized_img = convert_to_grayscale(img_bgr)

    # 4.创建处理前后对比图
    processed_images = [img_gray, img_blur, equalized_img]
    titles = ["Original Image", "Grayscale", "Gaussian Blur", "Histogram Equalization"]
    comparison_img = create_processing_comparison(img_bgr, processed_images, titles)
    
    # 5. 保存图像 
    img_name = os.path.basename(image_path)
    save_images(img_bgr, img_gray, img_blur, equalized_img, comparison_img, img_name, save_dir=output_dir)

if __name__ == "__main__":
    run_preprocessing_pipeline()