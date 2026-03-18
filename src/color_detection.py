import cv2
import numpy as np
import sys
import os

COLOR_RANGES = {
    "red": {
        "lower1": np.array([0, 100, 100]),    # 红色下限1
        "upper1": np.array([10, 255, 255]),   # 红色上限1
        "lower2": np.array([160, 100, 100]),  # 红色下限2（跨越0度）
        "upper2": np.array([179, 255, 255]),  # 红色上限2
        "color": (0, 0, 255)                   # BGR显示颜色
    },
    "blue": {
        "lower": np.array([100, 100, 100]),   # 蓝色下限
        "upper": np.array([130, 255, 255]),   # 蓝色上限
        "color": (255, 0, 0)                   # BGR显示颜色
    },
    "green": {
        "lower": np.array([40, 50, 50]),      # 绿色下限
        "upper": np.array([80, 255, 255]),    # 绿色上限
        "color": (0, 255, 0)                   # BGR显示颜色
    }
}

def create_color_mask(hsv_image, color_name):
    """
    根据颜色名称创建HSV颜色掩码

    参数:
        hsv_image: np.ndarray - 转换为HSV色彩空间的图像
        color_name: str - 目标颜色名称（支持red/blue/green）
    返回:
        np.ndarray | None - 二值化颜色掩码（白色为目标区域），不支持的颜色返回None
    """
    color_info = COLOR_RANGES.get(color_name)
    if color_info is None:
        print(f"[错误] 不支持的颜色: {color_name}")
        return None
    
    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    if color_name == "red":
     temp_mask1 = cv2.inRange(hsv_image, color_info["lower1"], color_info["upper1"])
     temp_mask2 = cv2.inRange(hsv_image, color_info["lower2"], color_info["upper2"])
     mask = cv2.bitwise_or(temp_mask1, temp_mask2)

    elif color_name == "blue":
        lower = color_info["lower"]
        upper = color_info["upper"]
        mask = cv2.inRange(hsv_image, lower, upper)
    return mask

def apply_morphology(mask, kernel_size=5, operations=["open", "close"]):
    """
    对二值掩码应用形态学操作去噪和优化

    参数:
        mask: np.ndarray - 原始二值化掩码图像
        kernel_size: int - 形态学操作核的尺寸，默认5
        operations: list - 执行的形态学操作列表，默认["open", "close"]
            支持操作：open(开运算)、close(闭运算)、erode(腐蚀)、dilate(膨胀)
    返回:
        np.ndarray - 经过形态学处理后的掩码图像
    """
    # 统一使用椭圆核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    result = mask.copy()
    
    for op in operations:
        if op == "open":
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        elif op == "close":
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        elif op == "erode":
            result = cv2.erode(result, kernel, iterations=1)
        elif op == "dilate":
            result = cv2.dilate(result, kernel, iterations=1)
    
    return result

def find_contours(mask, min_area=100, max_area=None):
    """
    从掩码中查找有效轮廓并计算轮廓关键信息

    参数:
        mask: np.ndarray - 去噪后的二值化掩码图像
        min_area: int - 最小轮廓面积阈值，过滤小噪声，默认100像素
        max_area: int | None - 最大轮廓面积阈值，超出则过滤，默认None（不限制）
    返回:
        tuple - (valid_contours, contour_info)
            valid_contours: list - 符合面积条件的轮廓列表
            contour_info: list - 每个轮廓的信息字典，包含：
                id: 轮廓编号
                center: 轮廓中心点坐标 (cx, cy)
                area: 轮廓面积
                bbox: 外接矩形 (x, y, w, h)
                circle_center: 最小外接圆圆心 (x, y)
                circle_radius: 最小外接圆半径
    备注:
        轮廓会按面积降序排序，优先保留大面积目标
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    contour_info = []
    
    for i, contour in enumerate(contours):
        area = round(cv2.contourArea(contour), 2)
        
        # 面积筛选
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        
        # 矩计算中心点
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = 0, 0
        
        # 外接矩形
        x, y, w, h = cv2.boundingRect(contour)

        # 外接圆
        (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
        valid_contours.append(contour)
        contour_info.append({
            "id": i + 1,
            "center": (cx, cy),
            "area": area,
            "bbox": (x, y, w, h),
            "circle_center": (int(circle_x), int(circle_y)),
            "circle_radius": int(radius)
        }) 
    # 按面积降序排序
    sorted_pairs = sorted(
        zip(valid_contours, contour_info), 
        key=lambda x: x[1]["area"], 
        reverse=True
    ) 
    if sorted_pairs:
        valid_contours, contour_info = zip(*sorted_pairs)
        valid_contours = list(valid_contours)
        contour_info = list(contour_info)
    
    return valid_contours, contour_info

def draw_contour_info(image, contours, contour_info, color=(0, 255, 0)):
    """
    在图像上绘制轮廓及相关信息（中心点、外接矩形、文本标注）

    参数:
        image: np.ndarray - 原始BGR图像
        contours: list - 待绘制的轮廓列表
        contour_info: list - 轮廓对应的信息字典列表
        color: tuple - 绘制轮廓的颜色（BGR格式），默认(0, 255, 0)
    返回:
        np.ndarray - 标注后的图像副本（不修改原始图像）
    备注:
        会自动处理空轮廓/信息不匹配等异常情况，避免程序报错
    """
    # 1. 复制原图，避免直接修改输入
    annotated = image.copy()
    
    # 2.避免报错
    if not isinstance(contours, list) or len(contours) == 0:
        return annotated
    if not isinstance(contour_info, list) or len(contour_info) == 0:
        return annotated
    if len(contours) != len(contour_info):
        return annotated
    
    # 3. 遍历轮廓和信息
    for i, (contour, info) in enumerate(zip(contours, contour_info)):
        contour_id = info.get("id", i + 1)  # 无id时用索引+1
        center = info.get("center", (0, 0)) # 无center时默认(0,0)
        cx, cy = center
        area = info.get("area", 0)          # 无area时默认0
        bbox = info.get("bbox", (0, 0, 0, 0)) # 无bbox时默认(0,0,0,0)
        x, y, w, h = bbox
        
        # 1. 绘制轮廓
        cv2.drawContours(annotated, [contour], -1, color, 2)
        
        # 2. 绘制中心点
        cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
        
        # 3. 绘制外接矩形
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)    

        text = f"ID:{contour_id} ({cx},{cy}) A:{area:.2f}"

        cv2.putText(
            annotated, 
            text, 
            (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            color, 
            2
        )
    
    return annotated

def create_color_comparison(original, detections):
    """
    创建颜色检测综合对比图
    
    参数:
        original: np.ndarray - 原始BGR图像
        detections: dict - 检测结果字典（key=颜色名，value={mask_clean:掩码, annotated:标注图}）
    返回:
        np.ndarray - 拼接后的综合对比图
    """
    rows = []
    
    # 第一行：原始图 + 各颜色掩码
    masks = [original]
    mask_titles = ["Original"]
    
    for color_name, data in detections.items():
        mask_display = cv2.cvtColor(data["mask_clean"], cv2.COLOR_GRAY2BGR)
        masks.append(mask_display)
        mask_titles.append(f"{color_name.capitalize()} Mask")
    
    # 填充空白使每行数量一致
    while len(masks) < 4:
        masks.append(np.zeros_like(original))
        mask_titles.append("")
    
    # 统一尺寸并拼接第一行
    height, width = original.shape[:2]
    mask_row = []
    for img in masks[:4]:
        mask_row.append(cv2.resize(img, (width, height)))
    rows.append(np.hstack(mask_row))
    # 第二行：原始图 + 各颜色检测标注结果
    annotated_images = [original]
    annotated_titles = ["Original"]
    for color_name, data in detections.items():
        annotated_images.append(data["annotated"])
        annotated_titles.append(f"{color_name.capitalize()} Detection")
    # 填充空白使每行数量一致
    while len(annotated_images) < 4:
        annotated_images.append(np.zeros_like(original))
        annotated_titles.append("")
    # 统一尺寸并拼接第二行
    annotated_row = []
    for img in annotated_images[:4]:
        annotated_row.append(cv2.resize(img, (width, height)))
    rows.append(np.hstack(annotated_row))
    
    # 垂直拼接两行得到最终对比图
    comparison = np.vstack(rows)
    
    # 添加标题（黑边白字提升可读性）
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # 第一行标题
    for i, title in enumerate(mask_titles[:4]):
        if title:
            x = i * width + 10
            y = 30
            # 黑色描边
            cv2.putText(comparison, title, (x, y), font, font_scale, (0, 0, 0), thickness + 1)
            # 白色字体
            cv2.putText(comparison, title, (x, y), font, font_scale, (255, 255, 255), thickness)
    
    # 第二行标题
    row2_offset = height
    for i, title in enumerate(annotated_titles[:4]):
        if title:
            x = i * width + 10
            y = row2_offset + 30
            # 黑色描边
            cv2.putText(comparison, title, (x, y), font, font_scale, (0, 0, 0), thickness + 1)
            # 白色字体
            cv2.putText(comparison, title, (x, y), font, font_scale, (255, 255, 255), thickness)
    
    return comparison

def detect_color_targets(image, target_colors=["red", "blue"], output_dir="output"):
    """
    执行多颜色目标检测主流程

    参数:
        image: np.ndarray - 输入的原始BGR图像
        target_colors: list - 待检测的颜色列表，默认["red", "blue"]
        output_dir: str - 检测结果保存目录，默认"output"
    返回:
        dict - 检测结果字典，key为颜色名称，value为该颜色的检测结果：
            mask_clean: np.ndarray - 去噪后的颜色掩码
            annotated: np.ndarray - 该颜色目标的标注图像
            info: list - 轮廓信息字典列表
    """
    
    # 转换为HSV空间（颜色检测的核心步骤）
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    results = {}
    
    for color_name in target_colors:
        # 1. 创建颜色掩码
        mask = create_color_mask(hsv, color_name)
        if mask is None:
            continue
        
        # 2. 形态学去噪
        denoised_mask = apply_morphology(mask)
        
        # 3. 查找有效轮廓
        contours, contour_info = find_contours(denoised_mask)
        
        # 4. 绘制该颜色的标注图
        annotated = draw_contour_info(image, contours, contour_info, COLOR_RANGES[color_name]["color"])
        
        # 5. 构建该颜色的检测结果（适配对比图函数）
        results[color_name] = {
            "mask_clean": denoised_mask,    # 去噪后的掩码
            "annotated": annotated,      # 该颜色的标注图
            "info": contour_info         # 轮廓信息（保留用于打印结果）
        }
        
        # 保存单独的掩码和标注图
        cv2.imwrite(os.path.join(output_dir, f"{color_name}_mask.jpg"),denoised_mask)
        cv2.imwrite(os.path.join(output_dir, f"{color_name}_detection.jpg"), annotated)
    
    # 生成并保存全颜色标注图
    full_annotated = image.copy()
    for color_name, data in results.items():
        contours, _ = find_contours(data["mask_clean"])
        full_annotated = draw_contour_info(
            full_annotated, contours, data["info"], COLOR_RANGES[color_name]["color"]
        )
    cv2.imwrite(os.path.join(output_dir, "full_annotated_result.jpg"), full_annotated)
    return results

def color_detection_main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "images/color_test.jpg"
    
    # 2. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像，请检查路径是否正确")
    
    # 3. 执行检测
    detection_results = detect_color_targets(img)
    
    # 4. 生成并保存对比图
    comparison_img = create_color_comparison(img, detection_results)
    cv2.imwrite("output/color_comparison.jpg", comparison_img)
    
    # 5. 打印检测结果
    print("="*50)
    print("颜色检测结果：")
    for color, data in detection_results.items():
        info_list = data["info"]
        print(f"\n{color.upper()} 目标数量: {len(info_list)}")
        for info in info_list:
            print(f"  目标:{info['id']} 中心坐标:{info['center']} 面积:{info['area']:.2f}")
if __name__ == "__main__":
   color_detection_main()