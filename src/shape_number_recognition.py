
import cv2
import numpy as np
import os
import sys
from basic_preprocessing import convert_to_grayscale

def get_shape_color(shape_type):
    """
    为不同几何图形分配标注颜色（BGR格式）
    :param shape_type: 形状名称字符串
    :return: BGR颜色元组
    """
    color_map = {
        "Circle": (0, 0, 255),        # 圆形-红色
        "Square": (0, 255, 0),        # 正方形-绿色
        "Rectangle": (255, 0, 0),     # 矩形-蓝色
        "Triangle": (0, 255, 255),    # 三角形-黄色
        "Pentagon": (255, 0, 255),    # 五边形-品红
        "Hexagon": (255, 255, 0),     # 六边形-青色
        "Ellipse": (128, 0, 128),     # 椭圆-紫色
        "Polygon": (128, 128, 128)    # 多边形-灰色
    }
    return color_map.get(shape_type, (0, 0, 0))  # 默认黑色

def preprocess_for_recognition(image):
    """
    预处理图像用于识别（复用basic_preprocessing现有函数）
    
    参数:
        image: 输入图像（BGR格式彩色图 / 灰度图）
    返回:
        binary: 二值化图像（目标为白色）
        edges: Canny边缘检测图像
    """
    # 复用basic_preprocessing的灰度转换+高斯模糊
    if len(image.shape) == 3:  
        gray, blurred, _ = convert_to_grayscale(image)
    else: 
        gray = image.copy()
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  
    
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2 )
    
    # Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    
    return binary, edges

def classify_shape(contour):
    """
    根据轮廓特征分类几何图形
    :param contour: 单个轮廓
    :return: shape_type（形状名称）, center（轮廓中心点）
    """
    shape_type = "Unknown"
    # 计算轮廓面积和周长
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True) if area > 0 else 0
    
    # 计算轮廓中心点
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        center = (cX, cY)
    else:
        center = (0, 0)
    
   # 面积过滤
    if area < 100:
        return shape_type, center
    
    # 多边形逼近（减少顶点数，突出形状特征）
    epsilon = 0.04 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertex_count = len(approx)
    
    # 计算圆形度（4π*面积/周长²，圆形接近1，其他形状<1）
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    
    # 形状分类规则
    if vertex_count == 3:
        shape_type = "Triangle"
    elif vertex_count == 4:
        # 计算宽高比判断正方形/矩形
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        shape_type = "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
    elif vertex_count == 5:
        shape_type = "Pentagon"
    elif vertex_count == 6:
        shape_type = "Hexagon"
    else:
        # 顶点数>6：判断圆形/椭圆/多边形
        if circularity > 0.75:
            shape_type = "Circle"
        elif 0.5 < circularity <= 0.75:
            shape_type = "Ellipse"
        else:
            shape_type = "Polygon"
    
    return shape_type, center

def detect_geometric_shapes(image, binary=None):
    """
    检测图像中的几何图形
    :param image: 原始BGR图像
    :param binary: 预处理后的二值图（None则自动处理）
    :return: shapes（检测结果列表）, annotated_image（标注后的图像）
    """
    # 预处理
    if binary is None:
        binary, _ = preprocess_for_recognition(image)
    
    annotated_image = image.copy()
    shapes = []
    
    # 形态学闭运算：填充轮廓内部小空隙
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 查找外部轮廓
    contours, _ = cv2.findContours(
        binary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 逐个轮廓分类并标注
    for cnt in contours:
        shape_type, center = classify_shape(cnt)
        if shape_type == "Unknown":
            continue
        
        # 计算轮廓面积和外接矩形
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 存储结果
        shapes.append({
            "type": shape_type,
            "center": center,
            "area": round(area, 1),
            "bbox": (x, y, w, h)
        })
        
        # 绘制标注：轮廓+中心点+形状名称
        color = get_shape_color(shape_type)
        cv2.drawContours(annotated_image, [cnt], -1, color, 2)
        cv2.circle(annotated_image, center, 5, color, -1)
        cv2.putText(
            annotated_image, shape_type, (center[0]-20, center[1]-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
    
    return shapes, annotated_image

def calculate_contour_similarity(cnt1, cnt2):
    """
    计算两个轮廓的相似度（用于判断子轮廓是否是父轮廓的孔洞）
    
    参数:
        cnt1: 父轮廓
        cnt2: 子轮廓（候选孔洞）
    返回:
        similarity_score: 0-1之间的相似度分数（越接近1越相似）
        is_valid_hole: 是否是有效的孔洞
    """
    # 1. 面积比例检查（孔洞面积应该明显小于父轮廓）
    area1 = cv2.contourArea(cnt1)
    area2 = cv2.contourArea(cnt2)
    
    if area1 <= 0 or area2 <= 0:
        return 0.0, False
    
    area_ratio = area2 / area1
    # 孔洞面积通常占父轮廓的3%-40%
    if not (0.03 <= area_ratio <= 0.4):
        return 0.0, False
    
    # 2. 形状相似度：使用Hu矩（旋转、缩放、平移不变）
    moments1 = cv2.moments(cnt1)
    moments2 = cv2.moments(cnt2)
    
    if moments1["m00"] == 0 or moments2["m00"] == 0:
        return 0.0, False
    
    # 计算Hu矩
    hu1 = cv2.HuMoments(moments1).flatten()
    hu2 = cv2.HuMoments(moments2).flatten()
    
    # 对Hu矩取对数（因为原始值范围差异大）
    hu1 = np.sign(hu1) * np.log10(np.abs(hu1) + 1e-10)
    hu2 = np.sign(hu2) * np.log10(np.abs(hu2) + 1e-10)
    
    # 计算欧氏距离并转换为相似度
    distance = np.linalg.norm(hu1 - hu2)
    hu_similarity = 1 / (1 + distance)  # 转换为0-1范围
    
    # 3. 圆形度相似度（孔洞和外形通常都接近圆形/椭圆）
    perimeter1 = cv2.arcLength(cnt1, True)
    perimeter2 = cv2.arcLength(cnt2, True)
    
    circularity1 = (4 * np.pi * area1) / (perimeter1 ** 2) if perimeter1 > 0 else 0
    circularity2 = (4 * np.pi * area2) / (perimeter2 ** 2) if perimeter2 > 0 else 0
    
    # 圆形度差异越小越相似
    circ_diff = abs(circularity1 - circularity2)
    circ_similarity = 1 - min(circ_diff, 1.0)
    
    # 4. 重心位置检查（孔洞应该在父轮廓内部且大致居中）
    M1 = cv2.moments(cnt1)
    M2 = cv2.moments(cnt2)
    
    if M1["m00"] > 0 and M2["m00"] > 0:
        cx1 = int(M1["m10"] / M1["m00"])
        cy1 = int(M1["m01"] / M1["m00"])
        cx2 = int(M2["m10"] / M2["m00"])
        cy2 = int(M2["m01"] / M2["m00"])
        
        # 获取父轮廓的边界框
        x1, y1, w1, h1 = cv2.boundingRect(cnt1)
        
        # 计算相对位置（孔洞重心应该在父轮廓内部）
        rel_x = (cx2 - x1) / float(w1) if w1 > 0 else 0
        rel_y = (cy2 - y1) / float(h1) if h1 > 0 else 0
        
        # 孔洞重心应该在父轮廓的中心区域（0.15-0.85范围内）
        if not (0.15 <= rel_x <= 0.85 and 0.15 <= rel_y <= 0.85):
            position_score = 0.0
        else:
            # 越接近中心分数越高
            center_dist = np.sqrt((rel_x - 0.5)**2 + (rel_y - 0.5)**2)
            position_score = 1 - min(center_dist * 2, 1.0)
    else:
        position_score = 0.0
    
    # 5. 综合相似度（加权平均）
    similarity_score = (
        0.3 * hu_similarity +      # Hu矩形状相似度
        0.3 * circ_similarity +     # 圆形度相似度
        0.4 * position_score        # 位置合理性（更重要）
    )
    
    # 判断是否为有效孔洞
    is_valid_hole = (
        similarity_score > 0.5 and  # 整体相似度阈值（降低要求）
        area_ratio > 0.03 and       # 面积比例下限
        position_score > 0.2        # 位置不能太偏（降低要求）
    )
    
    return similarity_score, is_valid_hole

def recognize_digits_contour_based(image, binary=None):
    """
    轮廓法数字识别（印刷体0-9）
    :param image: 原始BGR图像
    :param binary: 预处理后的二值图（None则自动处理）
    :return: results（识别结果列表）, annotated_image（标注后的图像）
    """
    # 1. 图像预处理
    if binary is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    annotated_image = image.copy()
    results = []

    # 2. 使用RETR_TREE获取完整轮廓层级
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if hierarchy is None or len(contours) == 0:
        return results, annotated_image
    
    hierarchy = hierarchy[0]
    
    # 3. 筛选数字轮廓（外层轮廓）
    digit_candidates = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        # 数字通常较小，面积在100-3000之间
        if area < 100 or area > 3000:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        # 数字的宽高比通常在0.2-1.2之间
        if aspect_ratio < 0.2 or aspect_ratio > 1.2:
            continue
        
        # 只保留无父轮廓的外层轮廓（数字主体）
        if hierarchy[i][3] == -1:
            # 计算轮廓的位置 - 数字通常在图像下方
            image_height = image.shape[0]
            # 只保留图像下方的轮廓（假设数字在下方）
            if y > image_height * 0.6:
                # 进一步验证：确保这是数字而不是噪声
                # 检查轮廓的紧凑度
                rect_area = w * h
                compactness = area / rect_area if rect_area > 0 else 0
                if compactness > 0.2:  # 降低紧凑度要求，使更多数字能够被识别
                    digit_candidates.append((i, cnt, (x, y, w, h)))
    
    # 按x坐标排序
    digit_candidates.sort(key=lambda x: x[2][0])
    
    # 4. 逐个数字识别
    for contour_idx, cnt, bbox in digit_candidates:
        x, y, w, h = bbox
        
        hole_count = 0
        hole_details = []  # 记录孔洞详情用于调试
        
        # 1. 检查所有子轮廓
        for j in range(len(contours)):
            # 检查是否是当前轮廓的子轮廓
            current_hier = hierarchy[j]
            parent_idx = current_hier[3]
            
            # 追踪父轮廓链，检查是否最终指向当前轮廓
            temp_parent = parent_idx
            while temp_parent != -1:
                if temp_parent == contour_idx:
                    # 找到一个子轮廓
                    child_cnt = contours[j]
                    
                    # 计算子轮廓与父轮廓的相似度
                    similarity, is_valid_hole = calculate_contour_similarity(cnt, child_cnt)
                    
                    if is_valid_hole:
                        hole_count += 1
                        hole_details.append({
                            "index": j,
                            "similarity": round(similarity, 3),
                        })
                    break
                temp_parent = hierarchy[temp_parent][3]
        
        # 2. 对每个数字ROI使用RETR_CCOMP重新检测孔洞
        roi = binary[y:y+h, x:x+w]
        roi_contours, roi_hier = cv2.findContours(
            roi.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if roi_hier is not None:
            ccomp_hole_count = 0
            for k in range(len(roi_contours)):
                if roi_hier[0][k][3] != -1:  # 子轮廓（孔洞）
                    hole_area = cv2.contourArea(roi_contours[k])
                    parent_area = cv2.contourArea(cnt)
                    if hole_area > 15 and (hole_area / parent_area) < 0.5:
                        ccomp_hole_count += 1
            
            # 使用两种方法的最大值作为最终孔洞数
            if ccomp_hole_count > hole_count:
                hole_count = ccomp_hole_count
        
        # 特征提取
        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.05 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        vertex_count = len(approx)
        
        aspect_ratio = w / float(h)
        contour_area = cv2.contourArea(cnt)
        rect_area = w * h
        compactness = contour_area / rect_area if rect_area > 0 else 0
        
        # 重心计算
        M = cv2.moments(cnt)
        rel_cy = 0.5
        if M["m00"] != 0:
            cy = int(M["m01"] / M["m00"])
            rel_cy = (cy - y) / float(h)
        
        # 数字分类 
        digit = -1
        confidence = 0.0
        
        if hole_count >= 2:
            digit = 8
            confidence = 0.95 if hole_count == 2 else 0.85
        elif hole_count == 1:
            if vertex_count < 8 and 0.8 <= aspect_ratio <= 1.2:
                digit = 0
                confidence = 0.9
            elif aspect_ratio < 0.85:
                # 6 vs 9：通过重心位置区分
                if rel_cy > 0.55:
                    digit = 6
                    confidence = 0.88
                else:
                    digit = 9
                    confidence = 0.88
            else:
                digit = 9
                confidence = 0.85
        elif hole_count == 0:
            # 数字1的特征：非常窄的矩形，顶点数少
            if aspect_ratio < 0.3 and vertex_count <= 3:
                digit = 1
                confidence = 0.95
            # 数字7的特征：较窄的矩形，顶点数2-4
            elif 0.3 <= aspect_ratio < 0.6 and vertex_count <= 4:
                digit = 7
                confidence = 0.9
            # 数字2的特征：顶点数4-6，紧凑度较高
            elif (vertex_count >= 4 and vertex_count <= 6) and compactness > 0.55:
                digit = 2
                confidence = 0.85
            # 数字5的特征：顶点数4-6，紧凑度较低
            elif (vertex_count >= 4 and vertex_count <= 6) and compactness <= 0.55:
                digit = 5
                confidence = 0.85
            # 数字4的特征：顶点数4，宽高比适中
            elif vertex_count == 4 and aspect_ratio < 0.65:
                digit = 4
                confidence = 0.85
            # 数字3的特征：顶点数7以上，宽高比大于0.6
            elif vertex_count >= 7 and aspect_ratio > 0.6:
                digit = 3
                confidence = 0.82
         
        # 置信度修正
        if digit != -1:
            confidence = min(confidence, 1.0)
        
        # 保存结果
        if digit != -1:
            results.append({
                "digit": digit,
                "confidence": round(confidence, 2),
                "bbox": (x, y, w, h),
                "features": {
                    "holes": hole_count,
                    "vertices": vertex_count,
                    "aspect_ratio": round(aspect_ratio, 2),
                    "hole_details": hole_details
                }
            })
            
            cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{digit} "
            cv2.putText(
                annotated_image, label,
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
    
    return results, annotated_image

def run_shape_recognition(output_dir="output", image_path=None):
    """
    运行完整的形状和数字识别流程
    
    参数:
        image_path: 输入图像路径
        output_dir: 输出目录
    """
    if image_path is None:
        # 使用默认测试图像
        image_path = "images\shape_number_test.jpg"
    image = cv2.imread(image_path)
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    if image is None:
        print(f"[WARN] 未指定图像路径，使用默认路径：{image_path}")
        print(f"错误：无法读取图像 {image_path}，请检查路径是否正确！")
        return
    
    print("[INFO] 开始执行形状+数字识别流程")
    print("="*60)
    
    binary, _ = preprocess_for_recognition(image)
    print("[INFO] 图像预处理完成")
    
    shapes_result, shapes_image = detect_geometric_shapes(image, binary)
    print("[INFO] 几何图形检测完成，共检测到 {} 个图形".format(len(shapes_result)))
    print("\n=== 几何图形检测结果 ===")
    for shape in shapes_result:
        print(f"形状：{shape['type']}，中心：{shape['center']}，面积：{shape['area']}")
    
    digits_result, digits_image = recognize_digits_contour_based(image, binary)
    print("\n=== 轮廓法数字识别结果 ===")
    print(f"[INFO] 轮廓法数字识别完成，共识别 {len(digits_result)} 个数字")
    for digit in digits_result:
        feat = digit.get('features', {})
        holes = feat.get('holes', 0)
        print(f"数字：{digit['digit']}，置信度：{digit['confidence']}，"
              f"孔洞：{holes}个，顶点：{feat.get('vertices', 'N/A')}")
       
    
    print("\n[INFO] 形状+数字识别流程执行完成")
    print(f"[INFO] 结果保存至：{output_dir}/")
    
    cv2.imwrite(f"{output_dir}/{base_name}_shape_detection_result.png", shapes_image)
    cv2.imwrite(f"{output_dir}/{base_name}_digit_recognition_result.png", digits_image)
    

if __name__ == "__main__":
    run_shape_recognition()