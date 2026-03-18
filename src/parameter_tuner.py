import cv2
import numpy as np
import sys


class ParameterTuner:
    """参数调优工具类，支持HSV颜色阈值、边缘检测、形态学操作、阈值处理的可视化调参"""

    def __init__(self):
        """初始化参数调优工具，设置默认窗口名、处理模式和各模式初始参数"""
        self.window_name = "Parameter Tuner"
        self.image = None
        self.original_image = None
        self.current_mode = "color"  # color, edge, morphology, threshold

        # 参数值存储
        self.params = {
            "color": {
                "h_min": 0,
                "h_max": 179,
                "s_min": 100,
                "s_max": 255,
                "v_min": 100,
                "v_max": 255
            },
            "edge": {
                "canny_low": 50,
                "canny_high": 150,
                "blur_kernel": 5,
                "sobel_ksize": 3
            },
            "morphology": {
                "kernel_size": 5,
                "operation": 0,  # 0:open, 1:close, 2:erode, 3:dilate
                "iterations": 1
            },
            "threshold": {
                "thresh_type": 0,  # 0:binary, 1:binary_inv, 2:trunc, 3:tozero, 4:tozero_inv
                "thresh_value": 127,
                "max_value": 255,
                "adaptive": 0  # 0:global, 1:mean, 2:gaussian
            }
        }

    def set_image(self, image):
        """
        设置待处理的输入图像

        参数:
            image: np.ndarray，cv2读取的BGR格式图像数组
        """
        self.image = image

    def create_trackbars(self, mode):
        """
        根据指定模式创建对应的OpenCV滑动条(Trackbar)，先销毁原有窗口再新建

        参数:
            mode: str，调参模式，可选值：'color'/'edge'/'morphology'/'threshold'
        """
        self.mode = mode

        # 清除之前的窗口
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow(self.window_name)

        # 创建新窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        if mode == "color":
            # 颜色阈值模式Trackbar
            cv2.createTrackbar("H Low", self.window_name, self.params["color"]["h_min"], 179, self._update_color)
            cv2.createTrackbar("S Low", self.window_name, self.params["color"]["s_min"], 255, self._update_color)
            cv2.createTrackbar("V Low", self.window_name, self.params["color"]["v_min"], 255, self._update_color)
            cv2.createTrackbar("H High", self.window_name, self.params["color"]["h_max"], 179, self._update_color)
            cv2.createTrackbar("S High", self.window_name, self.params["color"]["s_max"], 255, self._update_color)
            cv2.createTrackbar("V High", self.window_name, self.params["color"]["v_max"], 255, self._update_color)

        elif mode == "edge":
            # 边缘检测模式Trackbar
            cv2.createTrackbar("Blur Kernel", self.window_name, self.params["edge"]["blur_kernel"], 21, self._update_edge)
            cv2.createTrackbar("Canny Low", self.window_name, self.params["edge"]["canny_low"], 255, self._update_edge)
            cv2.createTrackbar("Canny High", self.window_name, self.params["edge"]["canny_high"], 255, self._update_edge)

        elif mode == "morphology":
            # 形态学操作模式Trackbar
            cv2.createTrackbar("Kernel Size", self.window_name, self.params["morphology"]["kernel_size"], 21, self._update_morphology)
            cv2.createTrackbar("Operation", self.window_name, self.params["morphology"]["operation"], 3, self._update_morphology)

        elif mode == "threshold":
            # 阈值处理模式Trackbar
            cv2.createTrackbar("Threshold", self.window_name, self.params["threshold"]["thresh_value"], 255, self._update_threshold)
            cv2.createTrackbar("Method", self.window_name, self.params["threshold"]["adaptive"], 2, self._update_threshold)
            cv2.createTrackbar("Block Size", self.window_name, 11, 49, self._update_threshold)
            cv2.createTrackbar("C", self.window_name, 2, 10, self._update_threshold)

    def _update_color(self, value):
        """
        颜色阈值模式的滑动条回调函数，更新HSV参数并刷新显示
        OpenCV回调要求必传value参数

        参数:
            value: int，滑动条拖动的数值
        """
        try:
            self.params["color"]["h_min"] = cv2.getTrackbarPos("H Low", self.window_name)
            self.params["color"]["s_min"] = cv2.getTrackbarPos("S Low", self.window_name)
            self.params["color"]["v_min"] = cv2.getTrackbarPos("V Low", self.window_name)
            self.params["color"]["h_max"] = cv2.getTrackbarPos("H High", self.window_name)
            self.params["color"]["s_max"] = cv2.getTrackbarPos("S High", self.window_name)
            self.params["color"]["v_max"] = cv2.getTrackbarPos("V High", self.window_name)
            self.refresh_display()
        except:
            # 忽略初始创建时的错误
            pass

    def _update_edge(self, value):
        """
        边缘检测模式的滑动条回调函数，更新边缘检测参数并刷新显示
        自动保证模糊核为奇数，OpenCV回调要求必传value参数

        参数:
            value: int，滑动条拖动的数值
        """
        try:
            # 确保模糊核为奇数
            blur_kernel = cv2.getTrackbarPos("Blur Kernel", self.window_name)
            blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
            if blur_kernel % 2 == 0:
                cv2.setTrackbarPos("Blur Kernel", self.window_name, blur_kernel)
            self.params["edge"]["blur_kernel"] = blur_kernel

            self.params["edge"]["canny_low"] = cv2.getTrackbarPos("Canny Low", self.window_name)
            self.params["edge"]["canny_high"] = cv2.getTrackbarPos("Canny High", self.window_name)
            self.refresh_display()
        except:
            # 忽略初始创建时的错误
            pass

    def _update_morphology(self, value):
        """
        形态学操作模式的滑动条回调函数，更新形态学参数并刷新显示
        自动保证核大小为奇数，OpenCV回调要求必传value参数
        参数:
            value: int，滑动条拖动的数值
        """
        try:
            # 确保核大小为奇数
            kernel_size = cv2.getTrackbarPos("Kernel Size", self.window_name)
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            if kernel_size % 2 == 0:
                cv2.setTrackbarPos("Kernel Size", self.window_name, kernel_size)
            self.params["morphology"]["kernel_size"] = kernel_size

            self.params["morphology"]["operation"] = cv2.getTrackbarPos("Operation", self.window_name)
            self.refresh_display()
        except:
            # 忽略初始创建时的错误
            pass

    def _update_threshold(self, value):
        """
        阈值处理模式的滑动条回调函数，更新阈值处理参数并刷新显示
        自动保证块大小为奇数，OpenCV回调要求必传value参数，此处无实际业务使用

        参数:
            value: int，滑动条拖动的数值
        """
        try:
            self.params["threshold"]["thresh_value"] = cv2.getTrackbarPos("Threshold", self.window_name)
            self.params["threshold"]["adaptive"] = cv2.getTrackbarPos("Method", self.window_name)

            # 确保块大小为奇数
            block_size = cv2.getTrackbarPos("Block Size", self.window_name)
            block_size = block_size if block_size % 2 == 1 else block_size + 1
            if block_size % 2 == 0:
                cv2.setTrackbarPos("Block Size", self.window_name, block_size)

            # 存储阈值处理的额外参数
            if not "block_size" in self.params["threshold"]:
                self.params["threshold"]["block_size"] = 11
            self.params["threshold"]["block_size"] = block_size

            if not "c" in self.params["threshold"]:
                self.params["threshold"]["c"] = 2
            self.params["threshold"]["c"] = cv2.getTrackbarPos("C", self.window_name)

            self.refresh_display()
        except:
            # 忽略初始创建时的错误
            pass

    def refresh_display(self):
        """
        刷新窗口显示，水平拼接原图和处理后图像，添加模式和参数文字说明
        若处理后图像为灰度图，自动转换为BGR格式保证拼接维度一致
        """
        if self.image is None:
            return

        # 处理图像
        processed = self.process_image()

        # 水平拼接原图和处理后的图像
        if len(processed.shape) == 2:
            # 如果处理后的图像是灰度图，转换为彩色
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        combined = np.hstack((self.image, processed))

        # 添加文字信息
        cv2.putText(combined, f"Mode: {self.mode}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 添加对应模式的参数信息
        if self.mode == "color":
            param_text = f"HSV Low: ({self.params['color']['h_min']}, {self.params['color']['s_min']}, {self.params['color']['v_min']}) "
            param_text += f"High: ({self.params['color']['h_max']}, {self.params['color']['s_max']}, {self.params['color']['v_max']})"
            cv2.putText(combined, param_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        elif self.mode == "edge":
            param_text = f"Blur Kernel: {self.params['edge']['blur_kernel']} "
            param_text += f"Canny: ({self.params['edge']['canny_low']}, {self.params['edge']['canny_high']})"
            cv2.putText(combined, param_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        elif self.mode == "morphology":
            operations = ["Opening", "Closing", "Erosion", "Dilation"]
            op_name = operations[self.params['morphology']['operation']] if self.params['morphology']['operation'] < 4 else "Unknown"
            param_text = f"Kernel Size: {self.params['morphology']['kernel_size']} Operation: {op_name}"
            cv2.putText(combined, param_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        elif self.mode == "threshold":
            methods = ["Global", "Adaptive Mean", "Adaptive Gaussian"]
            method_name = methods[self.params['threshold']['adaptive']] if self.params['threshold']['adaptive'] < 3 else "Unknown"
            block_size = self.params['threshold'].get('block_size', 11)
            c = self.params['threshold'].get('c', 2)
            param_text = f"Threshold: {self.params['threshold']['thresh_value']} Method: {method_name} "
            param_text += f"Block: {block_size} C: {c}"
            cv2.putText(combined, param_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # 显示结果
        cv2.imshow(self.window_name, combined)

    def process_image(self):
        """
        根据当前设置的调参模式，调用对应处理函数处理图像

        返回:
            np.ndarray，处理后的图像数组（灰度/BGR格式，依模式而定）
        """
        if self.image is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        if self.mode == "color":
            return self.process_color()
        elif self.mode == "edge":
            return self.process_edge()
        elif self.mode == "morphology":
            return self.process_morphology()
        elif self.mode == "threshold":
            return self.process_threshold()
        else:
            return self.image

    def process_color(self):
        """
        颜色阈值处理：将图像转换为HSV格式，根据当前HSV参数生成掩码并提取目标颜色区域

        返回:
            np.ndarray，BGR格式的颜色提取结果图像
        """
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lower = np.array([self.params["color"]["h_min"], self.params["color"]["s_min"], self.params["color"]["v_min"]])
        upper = np.array([self.params["color"]["h_max"], self.params["color"]["s_max"], self.params["color"]["v_max"]])
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(self.image, self.image, mask=mask)
        return result

    def process_edge(self):
        """
        边缘检测处理：图像灰度化→高斯模糊→Canny边缘检测，提取图像边缘

        返回:
            np.ndarray，灰度格式的边缘检测结果图像
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur_kernel = self.params["edge"]["blur_kernel"]
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        edges = cv2.Canny(blurred, self.params["edge"]["canny_low"], self.params["edge"]["canny_high"])
        return edges

    def process_morphology(self):
        """
        形态学操作处理：图像灰度化→二值化→根据参数执行开/闭/腐蚀/膨胀操作

        返回:
            np.ndarray，灰度格式的形态学处理结果图像
        """
        # 灰度化
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # 二值化
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # 创建结构元素
        kernel_size = self.params["morphology"]["kernel_size"]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        # 执行形态学操作
        operation = self.params["morphology"]["operation"]
        if operation == 0:
            # 开运算
            result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        elif operation == 1:
            # 闭运算
            result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        elif operation == 2:
            # 腐蚀
            result = cv2.erode(binary, kernel)
        elif operation == 3:
            # 膨胀
            result = cv2.dilate(binary, kernel)
        else:
            result = binary

        return result

    def process_threshold(self):
        """
        阈值处理：根据参数执行全局阈值/自适应均值阈值/自适应高斯阈值处理

        返回:
            np.ndarray，灰度格式的阈值处理结果图像
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        adaptive = self.params["threshold"]["adaptive"]
        if adaptive == 0:
            # 全局阈值
            _, result = cv2.threshold(gray, self.params["threshold"]["thresh_value"], self.params["threshold"]["max_value"], cv2.THRESH_BINARY)
        elif adaptive == 1:
            # 均值自适应阈值
            block_size = self.params["threshold"].get("block_size", 11)
            c = self.params["threshold"].get("c", 2)
            result = cv2.adaptiveThreshold(gray, self.params["threshold"]["max_value"],
                                          cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY,
                                          block_size, c)
        elif adaptive == 2:
            # 高斯自适应阈值
            block_size = self.params["threshold"].get("block_size", 11)
            c = self.params["threshold"].get("c", 2)
            result = cv2.adaptiveThreshold(gray, self.params["threshold"]["max_value"],
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY,
                                          block_size, c)
        else:
            result = gray

        return result

    def save_parameters(self, filename=None):
        """
        将当前模式的调优参数保存到txt文件，自动创建output目录，默认文件名按模式生成

        参数:
            filename: str/None，自定义保存的文件名，为None时自动生成（params_模式.txt）

        返回:
            bool，保存成功返回True，失败返回False
        """
        # 确保 output 目录存在
        import os
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 生成文件名
        if filename is None:
            filename = f"params_{self.mode}.txt"

        # 完整文件路径
        full_path = os.path.join(output_dir, filename)

        # 构建参数文本
        lines = []
        lines.append(f"Mode: {self.mode}")
        lines.append("")
        lines.append("Parameters:")

        if self.mode == "color":
            lines.append(f"h_low: {self.params['color']['h_min']}")
            lines.append(f"s_low: {self.params['color']['s_min']}")
            lines.append(f"v_low: {self.params['color']['v_min']}")
            lines.append(f"h_high: {self.params['color']['h_max']}")
            lines.append(f"s_high: {self.params['color']['s_max']}")
            lines.append(f"v_high: {self.params['color']['v_max']}")
        elif self.mode == "edge":
            lines.append(f"blur_kernel: {self.params['edge']['blur_kernel']}")
            lines.append(f"canny_low: {self.params['edge']['canny_low']}")
            lines.append(f"canny_high: {self.params['edge']['canny_high']}")
        elif self.mode == "morphology":
            lines.append(f"morph_kernel: {self.params['morphology']['kernel_size']}")
            lines.append(f"morph_operation: {self.params['morphology']['operation']}")
        elif self.mode == "threshold":
            lines.append(f"threshold_value: {self.params['threshold']['thresh_value']}")
            lines.append(f"adaptive_method: {self.params['threshold']['adaptive']}")
            lines.append(f"block_size: {self.params['threshold'].get('block_size', 11)}")
            lines.append(f"c: {self.params['threshold'].get('c', 2)}")

        # 保存到文件
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print(f"参数已保存到: {full_path}")
            return True
        except Exception as e:
            print(f"保存参数失败: {str(e)}")
            return False

    def run(self, image_path):
        """
        启动参数调优工具的主方法，读取图像、创建滑动条、监听键盘事件

        参数:
            image_path: str，待处理图像的文件路径（相对/绝对路径均可）
        """
        # 读取图像
        self.image = cv2.imread(image_path)
        if self.image is None:
            print(f"无法读取图像: {image_path}")
            return

        # 创建Trackbar
        self.create_trackbars(self.mode)

        # 初始显示
        self.refresh_display()

        # 等待用户操作
        print("按ESC键退出调优，按S键保存参数")
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键
                break
            elif key == ord('s') or key == ord('S'):  # S键保存参数
                self.save_parameters()

        # 打印最终参数
        print("\n最终参数值:")
        if self.mode == "color":
            print(f"HSV Low: ({self.params['color']['h_min']}, {self.params['color']['s_min']}, {self.params['color']['v_min']})")
            print(f"HSV High: ({self.params['color']['h_max']}, {self.params['color']['s_max']}, {self.params['color']['v_max']})")
        elif self.mode == "edge":
            print(f"Blur Kernel: {self.params['edge']['blur_kernel']}")
            print(f"Canny Low: {self.params['edge']['canny_low']}")
            print(f"Canny High: {self.params['edge']['canny_high']}")
        elif self.mode == "morphology":
            operations = ["Opening", "Closing", "Erosion", "Dilation"]
            op_name = operations[self.params['morphology']['operation']] if self.params['morphology']['operation'] < 4 else "Unknown"
            print(f"Kernel Size: {self.params['morphology']['kernel_size']}")
            print(f"Operation: {op_name} ({self.params['morphology']['operation']})")
        elif self.mode == "threshold":
            methods = ["Global", "Adaptive Mean", "Adaptive Gaussian"]
            method_name = methods[self.params['threshold']['adaptive']] if self.params['threshold']['adaptive'] < 3 else "Unknown"
            block_size = self.params['threshold'].get('block_size', 11)
            c = self.params['threshold'].get('c', 2)
            print(f"Threshold: {self.params['threshold']['thresh_value']}")
            print(f"Method: {method_name} ({self.params['threshold']['adaptive']})")
            print(f"Block Size: {block_size}")
            print(f"C: {c}")

        # 清理
        cv2.destroyAllWindows()


def parse_arguments():
    """
    解析命令行参数，校验参数数量和模式有效性，无效则打印用法并退出程序

    返回:
        tuple，(mode, image_path)，有效调参模式和图像路径
    """
    if len(sys.argv) != 3:
        print("用法: python parameter_tuner.py <模式> <图像路径>")
        print("模式选项: color, edge, morphology, threshold")
        sys.exit(1)

    mode = sys.argv[1]
    image_path = sys.argv[2]

    # 验证模式是否有效
    valid_modes = ['color', 'edge', 'morphology', 'threshold']
    if mode not in valid_modes:
        print(f"无效的模式: {mode}")
        print(f"有效模式: {', '.join(valid_modes)}")
        sys.exit(1)

    return mode, image_path


if __name__ == "__main__":
    # 解析命令行参数
    mode, image_path = parse_arguments()

    # 创建调优工具实例
    tuner = ParameterTuner()
    # 设置调优模式
    tuner.mode = mode
    # 运行调优工具
    tuner.run(image_path)