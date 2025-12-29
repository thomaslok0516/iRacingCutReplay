"""
iRacing 视频剪辑工具
功能：从录制的视频中移除包含"REPLAY"字样的片段，保留纯净的游戏画面
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
import os
import subprocess
import shutil
import time
from typing import List, Tuple
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from pathlib import Path


class VideoCutter:
    """视频剪辑器"""
    
    def __init__(self):
        # OCR配置 - 识别大小写字母和数字（支持Replay、REPLAY、replay等）
        self.tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=REPLAYreplay0123456789'
        self.check_interval = 1  # 每1秒检测一帧（可调整）
        self.roi = None  # ROI区域 (x, y, width, height)，None表示使用全帧
        self.detection_mode = "fast"  # 检测模式: "fast" 或 "comprehensive"
        
    def set_roi(self, x: int, y: int, width: int, height: int):
        """
        设置ROI区域
        :param x: 左上角X坐标
        :param y: 左上角Y坐标
        :param width: 宽度
        :param height: 高度
        """
        self.roi = (x, y, width, height)
        print(f"设置ROI区域: x={x}, y={y}, width={width}, height={height}")
    
    def clear_roi(self):
        """清除ROI设置，使用全帧检测"""
        self.roi = None
        print("已清除ROI设置，使用全帧检测")
    
    def extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """
        从帧中提取ROI区域
        :param frame: 原始帧
        :return: ROI区域或原帧
        """
        if self.roi is None:
            return frame
        
        x, y, w, h = self.roi
        height, width = frame.shape[:2]
        
        # 确保ROI在帧范围内
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)
        
        if w <= 0 or h <= 0:
            return frame
        
        return frame[y:y+h, x:x+w]
    
    def preprocess_image(self, frame: np.ndarray, scale_factor: float = 4.0) -> List[np.ndarray]:
        """
        预处理图像 - 返回3个有效的预处理方法
        返回列表：
        [0] 方法3: OTSU二值化
        [1] 方法7: 反转OTSU
        [2] 方法10: 锐化
        :param frame: 原始图像
        :param scale_factor: 放大倍数
        :return: 预处理后的图像列表
        """
        results = []
        
        # 转换为灰度图
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # 放大图像（如果太小）
        h, w = gray.shape
        if h < 100 or w < 300:
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            gray_large = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            gray_large = gray
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_large)
        
        # 形态学操作：先膨胀再腐蚀，增强文字
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(enhanced, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        # 降噪
        denoised = cv2.fastNlMeansDenoising(eroded, None, 10, 7, 21)
        
        # 方法3: OTSU二值化
        _, binary_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(binary_otsu)
        
        # 方法7: 反转OTSU（黑底白字）
        results.append(cv2.bitwise_not(binary_otsu))
        
        # 方法10: 锐化
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
        _, binary_sharp = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(binary_sharp)
        
        return results
    
    def _try_ocr_detection(self, pil_image: Image.Image, psm: int) -> bool:
        """
        尝试OCR检测的辅助函数 - 优化版：只要检测到REPLAY就立即返回
        :param pil_image: PIL图像
        :param psm: PSM模式
        :return: 是否检测到REPLAY
        """
        try:
            # 包含大小写字母的白名单
            config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=REPLAYreplay0123456789'
            text = pytesseract.image_to_string(pil_image, config=config)
            
            # 快速检查：只要包含REPLAY就返回（不区分大小写）
            text_upper = text.upper()
            if 'REPLAY' in text_upper:
                return True
            
            # 检查原始文本（支持大小写混合）
            if 'Replay' in text or 'replay' in text.lower():
                return True
            
            # 如果都没有，返回False（不再进行部分匹配，提高速度）
            return False
        except Exception:
            return False
        
    def detect_replay_text(self, frame: np.ndarray) -> bool:
        """
        检测帧中是否包含"REPLAY"文字
        :param frame: 视频帧
        :return: 是否检测到REPLAY
        """
        try:
            # 如果设置了ROI，先提取ROI区域
            if self.roi is not None:
                frame = self.extract_roi(frame)
                original_roi = frame.copy()
            else:
                original_roi = None
            
            # 预处理图像，得到3个有效的预处理方法
            processed_images = self.preprocess_image(frame)
            
            # processed_images[0] = 方法3 (OTSU二值化)
            # processed_images[1] = 方法7 (反转OTSU)
            # processed_images[2] = 方法10 (锐化)
            
            if self.detection_mode == "fast":
                # 快速模式：只使用方法3 + PSM 6, 7
                if len(processed_images) > 0:
                    pil_image = Image.fromarray(processed_images[0])
                    for psm in [6, 7]:
                        if self._try_ocr_detection(pil_image, psm):
                            return True
                return False
            else:
                # 全面模式：Pipeline方式，按顺序尝试每个预处理方法 + PSM 6, 7, 11
                psm_modes = [6, 7, 11]  # 有效的PSM模式
                
                # Pipeline 1: 预处理方法3 + PSM 6, 7, 11
                if len(processed_images) > 0:
                    pil_image = Image.fromarray(processed_images[0])
                    for psm in psm_modes:
                        if self._try_ocr_detection(pil_image, psm):
                            return True
                
                # Pipeline 2: 预处理方法7 + PSM 6, 7, 11
                if len(processed_images) > 1:
                    pil_image = Image.fromarray(processed_images[1])
                    for psm in psm_modes:
                        if self._try_ocr_detection(pil_image, psm):
                            return True
                
                # Pipeline 3: 预处理方法10 + PSM 6, 7, 11
                if len(processed_images) > 2:
                    pil_image = Image.fromarray(processed_images[2])
                    for psm in psm_modes:
                        if self._try_ocr_detection(pil_image, psm):
                            return True
                
                return False
        except Exception as e:
            print(f"OCR识别错误: {e}")
            return False
    
    def analyze_video(self, video_path: str, progress_callback=None) -> List[Tuple[float, float]]:
        """
        分析视频，找出没有REPLAY的片段
        :param video_path: 视频路径
        :param progress_callback: 进度回调函数
        :return: 片段列表 [(开始时间, 结束时间), ...]
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"视频信息: FPS={fps}, 总帧数={total_frames}, 时长={duration:.2f}秒")
        
        # 片段列表
        segments = []
        current_segment_start = None
        last_check_time = 0
        
        frame_skip = max(1, int(fps * self.check_interval))  # 跳过的帧数
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            
            # 每隔一定时间检测一次
            if current_time - last_check_time >= self.check_interval:
                has_replay = self.detect_replay_text(frame)
                
                if not has_replay:
                    # 没有REPLAY，开始或继续片段
                    if current_segment_start is None:
                        current_segment_start = current_time
                else:
                    # 有REPLAY，结束当前片段
                    if current_segment_start is not None:
                        segments.append((current_segment_start, current_time))
                        current_segment_start = None
                
                last_check_time = current_time
                
                # 更新进度
                if progress_callback:
                    progress = (frame_count / total_frames) * 100
                    progress_callback(progress, f"分析中: {current_time:.1f}/{duration:.1f}秒")
            
            # 跳帧
            frame_count += frame_skip
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            
            # 检查是否超出范围
            if frame_count >= total_frames:
                break
        
        # 处理最后一个片段
        if current_segment_start is not None:
            segments.append((current_segment_start, duration))
        
        cap.release()
        return segments
    
    def _find_ffmpeg(self):
        """查找ffmpeg可执行文件"""
        # 首先检查系统PATH中是否有ffmpeg
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            return ffmpeg_path
        
        # 检查当前工作目录的上级目录（项目根目录）
        current_dir = os.getcwd()
        project_ffmpeg = os.path.join(current_dir, 'ffmpeg-8.0.1-essentials_build', 'bin', 'ffmpeg.exe')
        if os.path.exists(project_ffmpeg):
            return project_ffmpeg
        
        # 检查当前工作目录
        cwd_ffmpeg = os.path.join(current_dir, 'ffmpeg-8.0.1-essentials_build', 'bin', 'ffmpeg.exe')
        if os.path.exists(cwd_ffmpeg):
            return cwd_ffmpeg
        
        # 检查脚本所在目录的上级目录
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            script_ffmpeg = os.path.join(parent_dir, 'ffmpeg-8.0.1-essentials_build', 'bin', 'ffmpeg.exe')
            if os.path.exists(script_ffmpeg):
                return script_ffmpeg
        except:
            pass
        
        return None
    
    def cut_video(self, video_path: str, segments: List[Tuple[float, float]], 
                  output_folder: str, file_prefix: str = "race", progress_callback=None):
        """
        根据片段列表剪辑视频，为每个片段生成单独的文件（使用ffmpeg保留音频）
        :param video_path: 输入视频路径
        :param segments: 片段列表
        :param output_folder: 输出文件夹路径
        :param file_prefix: 文件前缀（例如: "race" → race_001.mp4）
        :param progress_callback: 进度回调函数
        :return: 生成的文件列表
        """
        if not segments:
            raise ValueError("没有找到可保留的片段")
        
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
        
        # 查找ffmpeg
        ffmpeg_path = self._find_ffmpeg()
        if not ffmpeg_path:
            raise ValueError("未找到ffmpeg，请确保ffmpeg已安装并在PATH中，或放在项目目录中")
        
        # 验证输入视频文件
        if not os.path.exists(video_path):
            raise ValueError(f"输入视频文件不存在: {video_path}")
        
        total_segments = len(segments)
        output_files = []
        
        for seg_idx, (start_time, end_time) in enumerate(segments):
            # 生成文件名：前缀_序号.mp4（例如: race_001.mp4）
            segment_num = seg_idx + 1
            filename = f"{file_prefix}_{segment_num:03d}.mp4"
            output_path = os.path.join(output_folder, filename)
            output_files.append(output_path)
            
            if progress_callback:
                progress = (seg_idx / total_segments) * 100
                progress_callback(progress, f"剪辑片段 {segment_num}/{total_segments}: {start_time:.1f}-{end_time:.1f}秒 → {filename}")
            
            # 计算片段时长
            duration = end_time - start_time
            
            # 使用ffmpeg剪辑视频（保留音频）
            # -ss: 开始时间
            # -t: 持续时间
            # -i: 输入文件
            # -c copy: 复制流（不重新编码，速度快）
            # -avoid_negative_ts make_zero: 避免负时间戳
            cmd = [
                ffmpeg_path,
                '-ss', str(start_time),
                '-i', video_path,
                '-t', str(duration),
                '-c', 'copy',  # 复制视频和音频流，不重新编码
                '-avoid_negative_ts', 'make_zero',
                '-y',  # 覆盖输出文件
                output_path
            ]
            
            try:
                # 运行ffmpeg命令（使用UTF-8编码，忽略编码错误）
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='ignore',  # 忽略编码错误
                    check=True
                )
            except subprocess.CalledProcessError as e:
                # 如果copy失败，尝试重新编码（更慢但更兼容）
                if progress_callback:
                    progress_callback(progress, f"使用copy模式失败，尝试重新编码...")
                
                cmd_reencode = [
                    ffmpeg_path,
                    '-ss', str(start_time),
                    '-i', video_path,
                    '-t', str(duration),
                    '-c:v', 'libx264',  # H.264视频编码
                    '-c:a', 'aac',      # AAC音频编码
                    '-preset', 'fast',  # 编码预设（平衡速度和质量）
                    '-crf', '23',       # 质量参数（18-28，23是默认值）
                    '-avoid_negative_ts', 'make_zero',
                    '-y',
                    output_path
                ]
                
                try:
                    result = subprocess.run(
                        cmd_reencode,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding='utf-8',
                        errors='ignore',  # 忽略编码错误
                        check=True
                    )
                except subprocess.CalledProcessError as e2:
                    # 安全地处理错误信息
                    error_msg = e2.stderr if isinstance(e2.stderr, str) else str(e2.stderr)
                    try:
                        error_msg = error_msg.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                    except:
                        error_msg = "ffmpeg处理失败（无法解析错误信息）"
                    raise ValueError(f"ffmpeg处理失败: {error_msg}")
        
        if progress_callback:
            progress_callback(100, f"剪辑完成！共生成 {total_segments} 个文件")
        
        return output_files


class VideoCutterGui:
    """GUI界面"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("iRacing 视频剪辑工具")
        self.root.geometry("800x900")
        self.root.minsize(750, 850)  # 设置最小尺寸
        
        self.video_path = None
        self.output_folder = None
        self.cutter = VideoCutter()
        self.segments = []
        self.analysis_time = 0.0  # 分析视频耗时（秒）
        self.cut_time = 0.0  # 剪辑视频耗时（秒）
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置UI界面"""
        # 标题
        title_label = tk.Label(
            self.root,
            text="iRacing 视频剪辑工具",
            font=("Arial", 18, "bold"),
            pady=10
        )
        title_label.pack()
        
        # 说明文字
        info_text = """
功能说明：
1. 选择已录制的视频文件
2. 程序会自动检测并移除包含"REPLAY"字样的片段
3. 保留纯净的游戏画面片段
4. 输出剪辑后的视频文件
        """
        info_label = tk.Label(
            self.root,
            text=info_text,
            justify=tk.LEFT,
            font=("Arial", 10),
            pady=5
        )
        info_label.pack()
        
        # 文件选择区域
        file_frame = tk.Frame(self.root, pady=10)
        file_frame.pack(fill=tk.X, padx=20)
        
        # 输入视频
        input_frame = tk.Frame(file_frame)
        input_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(input_frame, text="输入视频:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        input_path_frame = tk.Frame(input_frame)
        input_path_frame.pack(fill=tk.X, pady=5)
        
        self.input_path_var = tk.StringVar()
        input_entry = tk.Entry(input_path_frame, textvariable=self.input_path_var, 
                              font=("Arial", 10), state=tk.DISABLED)
        input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        input_btn = tk.Button(input_path_frame, text="选择文件", 
                             command=self.select_input_file, font=("Arial", 10))
        input_btn.pack(side=tk.RIGHT)
        
        # 输出设置
        output_frame = tk.Frame(file_frame)
        output_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(output_frame, text="输出设置:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        # 输出文件夹
        output_folder_frame = tk.Frame(output_frame)
        output_folder_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(output_folder_frame, text="保存文件夹:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.output_folder_var = tk.StringVar()
        output_folder_entry = tk.Entry(output_folder_frame, textvariable=self.output_folder_var,
                                       font=("Arial", 10), state=tk.DISABLED)
        output_folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        output_folder_btn = tk.Button(output_folder_frame, text="选择文件夹",
                                      command=self.select_output_folder, font=("Arial", 10))
        output_folder_btn.pack(side=tk.RIGHT)
        
        # 文件前缀
        output_prefix_frame = tk.Frame(output_frame)
        output_prefix_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(output_prefix_frame, text="文件前缀:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.output_prefix_var = tk.StringVar(value="race")
        output_prefix_entry = tk.Entry(output_prefix_frame, textvariable=self.output_prefix_var,
                                      font=("Arial", 10), width=20)
        output_prefix_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(output_prefix_frame, text="(例如: race → race_001.mp4, race_002.mp4)", 
                font=("Arial", 9), fg="gray").pack(side=tk.LEFT, padx=5)
        
        # 设置区域
        settings_frame = tk.Frame(self.root, pady=5)
        settings_frame.pack()
        
        interval_frame = tk.Frame(settings_frame)
        interval_frame.pack(pady=5)
        
        tk.Label(interval_frame, text="检测间隔(秒):", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.interval_var = tk.StringVar(value="1.0")
        interval_entry = tk.Entry(interval_frame, textvariable=self.interval_var, width=5, font=("Arial", 10))
        interval_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(interval_frame, text="(值越小检测越精确，但处理越慢)", 
                font=("Arial", 9), fg="gray").pack(side=tk.LEFT, padx=5)
        
        # 检测模式选择
        mode_frame = tk.Frame(settings_frame)
        mode_frame.pack(pady=5)
        
        tk.Label(mode_frame, text="检测模式:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.detection_mode_var = tk.StringVar(value="fast")  # 默认快速模式
        
        fast_mode_radio = tk.Radiobutton(
            mode_frame, 
            text="快速模式", 
            variable=self.detection_mode_var, 
            value="fast",
            font=("Arial", 10)
        )
        fast_mode_radio.pack(side=tk.LEFT, padx=10)
        
        comprehensive_mode_radio = tk.Radiobutton(
            mode_frame, 
            text="全面模式", 
            variable=self.detection_mode_var, 
            value="comprehensive",
            font=("Arial", 10)
        )
        comprehensive_mode_radio.pack(side=tk.LEFT, padx=10)
        
        mode_info_label = tk.Label(
            mode_frame, 
            text="快速: 方法3+PSM6,7 | 全面: 3方法×PSM6,7,11", 
            font=("Arial", 9), 
            fg="gray"
        )
        mode_info_label.pack(side=tk.LEFT, padx=10)
        
        # ROI区域设置
        roi_frame = tk.LabelFrame(settings_frame, text="ROI区域设置（可选，提高识别率）", 
                                  font=("Arial", 10, "bold"), pady=5, padx=10)
        roi_frame.pack(pady=5, padx=20, fill=tk.X)
        
        # ROI说明
        roi_info = tk.Label(roi_frame, 
                           text="如果REPLAY文字位置固定，可以指定区域以提高识别速度和准确率",
                           font=("Arial", 9), fg="gray", justify=tk.LEFT)
        roi_info.pack(anchor=tk.W, pady=(0, 5))
        
        # ROI输入框
        roi_input_frame = tk.Frame(roi_frame)
        roi_input_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(roi_input_frame, text="X:", font=("Arial", 10)).grid(row=0, column=0, padx=5)
        self.roi_x_var = tk.StringVar(value="600")  # 默认值
        tk.Entry(roi_input_frame, textvariable=self.roi_x_var, width=8, font=("Arial", 10)).grid(row=0, column=1, padx=5)
        
        tk.Label(roi_input_frame, text="Y:", font=("Arial", 10)).grid(row=0, column=2, padx=5)
        self.roi_y_var = tk.StringVar(value="20")  # 默认值
        tk.Entry(roi_input_frame, textvariable=self.roi_y_var, width=8, font=("Arial", 10)).grid(row=0, column=3, padx=5)
        
        tk.Label(roi_input_frame, text="宽度:", font=("Arial", 10)).grid(row=0, column=4, padx=5)
        self.roi_w_var = tk.StringVar(value="200")  # 默认值
        tk.Entry(roi_input_frame, textvariable=self.roi_w_var, width=8, font=("Arial", 10)).grid(row=0, column=5, padx=5)
        
        tk.Label(roi_input_frame, text="高度:", font=("Arial", 10)).grid(row=0, column=6, padx=5)
        self.roi_h_var = tk.StringVar(value="20")  # 默认值
        tk.Entry(roi_input_frame, textvariable=self.roi_h_var, width=8, font=("Arial", 10)).grid(row=0, column=7, padx=5)
        
        # ROI按钮
        roi_btn_frame = tk.Frame(roi_frame)
        roi_btn_frame.pack(pady=5)
        
        preview_btn = tk.Button(roi_btn_frame, text="预览视频帧", 
                               command=self.preview_video_frame, font=("Arial", 10))
        preview_btn.pack(side=tk.LEFT, padx=5)
        
        apply_roi_btn = tk.Button(roi_btn_frame, text="应用ROI", 
                                 command=self.apply_roi, font=("Arial", 10), bg="lightblue")
        apply_roi_btn.pack(side=tk.LEFT, padx=5)
        
        clear_roi_btn = tk.Button(roi_btn_frame, text="清除ROI（使用全帧）", 
                                 command=self.clear_roi, font=("Arial", 10))
        clear_roi_btn.pack(side=tk.LEFT, padx=5)
        
        # ROI状态显示
        self.roi_status_var = tk.StringVar(value="当前: 使用全帧检测")
        roi_status_label = tk.Label(roi_frame, textvariable=self.roi_status_var, 
                                   font=("Arial", 9), fg="blue")
        roi_status_label.pack(pady=5)
        
        # 进度条
        progress_frame = tk.Frame(self.root, pady=10)
        progress_frame.pack(fill=tk.X, padx=20)
        
        self.progress_var = tk.StringVar(value="等待开始...")
        self.progress_label = tk.Label(progress_frame, textvariable=self.progress_var,
                                      font=("Arial", 10))
        self.progress_label.pack(anchor=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=650)
        self.progress_bar.pack(fill=tk.X, pady=3)
        
        # 总时长显示
        time_info_frame = tk.Frame(progress_frame)
        time_info_frame.pack(fill=tk.X, pady=5)
        
        self.time_info_var = tk.StringVar(value="总操作时长: 0.0秒")
        self.time_info_label = tk.Label(time_info_frame, textvariable=self.time_info_var,
                                       font=("Arial", 9), fg="blue")
        self.time_info_label.pack(anchor=tk.W)
        
        # 片段信息
        segments_frame = tk.Frame(self.root)
        segments_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        tk.Label(segments_frame, text="检测到的片段:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        segments_text_frame = tk.Frame(segments_frame)
        segments_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.segments_text = tk.Text(segments_text_frame, height=4, font=("Consolas", 9), wrap=tk.WORD)  # 减少高度
        self.segments_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        segments_scrollbar = tk.Scrollbar(segments_text_frame)
        segments_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.segments_text.config(yscrollcommand=segments_scrollbar.set)
        segments_scrollbar.config(command=self.segments_text.yview)
        
        # 控制按钮（固定在底部，确保可见）
        button_frame = tk.Frame(self.root, pady=10)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20)  # 固定在底部
        
        self.analyze_btn = tk.Button(
            button_frame,
            text="分析视频",
            command=self.analyze_video,
            bg="blue",
            fg="white",
            font=("Arial", 12, "bold"),
            width=15,
            height=2
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=10)
        
        self.cut_btn = tk.Button(
            button_frame,
            text="开始剪辑",
            command=self.cut_video,
            bg="green",
            fg="white",
            font=("Arial", 12, "bold"),
            width=15,
            height=2,
            state=tk.DISABLED
        )
        self.cut_btn.pack(side=tk.LEFT, padx=10)
        
        # 一键分析和剪辑按钮
        self.auto_btn = tk.Button(
            button_frame,
            text="一键分析并剪辑",
            command=self.auto_analyze_and_cut,
            bg="orange",
            fg="white",
            font=("Arial", 12, "bold"),
            width=18,
            height=2
        )
        self.auto_btn.pack(side=tk.LEFT, padx=10)
        
        # 初始化时应用默认ROI
        self.apply_default_roi()
        
    def select_input_file(self):
        """选择输入视频文件"""
        filename = filedialog.askopenfilename(
            title="选择输入视频",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
        )
        if filename:
            self.video_path = filename
            self.input_path_var.set(filename)
            
    def select_output_folder(self):
        """选择输出文件夹"""
        folder = filedialog.askdirectory(title="选择输出文件夹")
        if folder:
            self.output_folder = folder
            self.output_folder_var.set(folder)
    
    def apply_default_roi(self):
        """应用默认ROI设置"""
        try:
            x = int(self.roi_x_var.get()) if self.roi_x_var.get() else 600
            y = int(self.roi_y_var.get()) if self.roi_y_var.get() else 20
            w = int(self.roi_w_var.get()) if self.roi_w_var.get() else 200
            h = int(self.roi_h_var.get()) if self.roi_h_var.get() else 20
            
            if w > 0 and h > 0 and x >= 0 and y >= 0:
                self.cutter.set_roi(x, y, w, h)
                self.roi_status_var.set(f"当前ROI: X={x}, Y={y}, 宽度={w}, 高度={h}")
        except:
            pass
    
    def auto_analyze_and_cut(self):
        """一键分析并剪辑视频"""
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showerror("错误", "请先选择输入视频文件")
            return
        
        if not self.output_folder:
            messagebox.showerror("错误", "请先选择输出文件夹")
            return
        
        # 应用当前ROI设置
        self.apply_roi()
        
        # 获取文件前缀
        file_prefix = self.output_prefix_var.get().strip()
        if not file_prefix:
            file_prefix = "race"
        
        # 获取检测间隔
        try:
            check_interval = float(self.interval_var.get())
            if check_interval <= 0:
                messagebox.showerror("错误", "检测间隔必须大于0")
                return
            self.cutter.check_interval = check_interval
        except ValueError:
            messagebox.showerror("错误", "检测间隔必须是数字")
            return
        
        # 设置检测模式
        self.cutter.detection_mode = self.detection_mode_var.get()
        
        # 重置时间（开始新的处理）
        self.analysis_time = 0.0
        self.cut_time = 0.0
        self.time_info_var.set("总操作时长: 0.0秒")
        
        # 禁用所有按钮
        self.analyze_btn.config(state=tk.DISABLED)
        self.cut_btn.config(state=tk.DISABLED)
        self.auto_btn.config(state=tk.DISABLED)
        
        def auto_thread():
            try:
                # 第一步：分析视频
                analysis_start_time = time.time()
                mode_name = "快速模式" if self.cutter.detection_mode == "fast" else "全面模式"
                self.update_progress(0, f"开始分析视频... ({mode_name})")
                segments = self.cutter.analyze_video(self.video_path, self.update_progress)
                self.segments = segments
                
                # 记录分析结束时间
                analysis_end_time = time.time()
                self.analysis_time = analysis_end_time - analysis_start_time
                
                if not segments:
                    messagebox.showwarning("提示", "未找到可保留的片段（所有片段都包含REPLAY字样）")
                    self.update_progress(0, f"分析完成，无片段可剪辑，耗时: {self.analysis_time:.2f}秒")
                    # 更新总时长显示
                    total_operation_time = self.analysis_time + self.cut_time
                    self.time_info_var.set(f"总操作时长: {total_operation_time:.2f}秒 (分析: {self.analysis_time:.2f}秒, 剪辑: {self.cut_time:.2f}秒)")
                    self.analyze_btn.config(state=tk.NORMAL)
                    self.cut_btn.config(state=tk.NORMAL)
                    self.auto_btn.config(state=tk.NORMAL)
                    return
                
                # 更新片段信息
                self.segments_text.delete(1.0, tk.END)
                total_time = sum(end - start for start, end in segments)
                self.segments_text.insert(tk.END, f"找到 {len(segments)} 个片段，总时长: {total_time:.2f} 秒\n\n")
                for i, (start, end) in enumerate(segments, 1):
                    self.segments_text.insert(tk.END, f"片段 {i}: {start:.2f}秒 - {end:.2f}秒 (时长: {end-start:.2f}秒)\n")
                
                self.update_progress(50, f"分析完成！找到 {len(segments)} 个片段，耗时: {self.analysis_time:.2f}秒，开始剪辑...")
                
                # 第二步：剪辑视频
                cut_start_time = time.time()
                output_files = self.cutter.cut_video(
                    self.video_path, 
                    segments, 
                    self.output_folder,
                    file_prefix,
                    self.update_progress
                )
                
                # 记录剪辑结束时间
                cut_end_time = time.time()
                self.cut_time = cut_end_time - cut_start_time
                
                # 计算总操作时长
                total_operation_time = self.analysis_time + self.cut_time
                
                # 更新总时长显示
                self.time_info_var.set(f"总操作时长: {total_operation_time:.2f}秒 (分析: {self.analysis_time:.2f}秒, 剪辑: {self.cut_time:.2f}秒)")
                
                # 生成文件列表消息
                files_info = "\n".join([os.path.basename(f) for f in output_files])
                messagebox.showinfo(
                    "成功", 
                    f"一键处理完成！\n\n共生成 {len(output_files)} 个文件：\n{files_info}\n\n保存位置: {self.output_folder}\n\n总操作时长: {total_operation_time:.2f}秒\n分析: {self.analysis_time:.2f}秒 | 剪辑: {self.cut_time:.2f}秒"
                )
                self.update_progress(100, f"一键处理完成！总耗时: {total_operation_time:.2f}秒")
                self.analyze_btn.config(state=tk.NORMAL)
                self.cut_btn.config(state=tk.NORMAL)
                self.auto_btn.config(state=tk.NORMAL)
                
            except Exception as e:
                messagebox.showerror("错误", f"处理失败: {e}")
                self.update_progress(0, "处理失败")
                self.analyze_btn.config(state=tk.NORMAL)
                self.cut_btn.config(state=tk.NORMAL)
                self.auto_btn.config(state=tk.NORMAL)
        
        threading.Thread(target=auto_thread, daemon=True).start()
    
    def preview_video_frame(self):
        """预览视频的第一帧，帮助确定ROI区域，支持ROI可视化"""
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showerror("错误", "请先选择输入视频文件")
            return
        
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                messagebox.showerror("错误", "无法打开视频文件")
                return
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                messagebox.showerror("错误", "无法读取视频帧")
                return
            
            # 保存原始帧
            original_frame = frame.copy()
            height, width = frame.shape[:2]
            
            # 创建一个新窗口显示预览
            preview_window = tk.Toplevel(self.root)
            preview_window.title("视频帧预览 - ROI可视化")
            preview_window.geometry("1100x800")
            preview_window.minsize(1000, 700)  # 设置最小尺寸
            
            # 显示视频信息
            info_text = f"视频尺寸: {width} x {height} 像素"
            info_label = tk.Label(preview_window, text=info_text, 
                                 font=("Arial", 10, "bold"), padx=20, pady=5)
            info_label.pack()
            
            # 创建主框架
            main_frame = tk.Frame(preview_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 左侧：预览图片
            left_frame = tk.Frame(main_frame)
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # 保存帧为临时图片
            temp_path = os.path.join(os.path.dirname(self.video_path), "temp_preview.jpg")
            cv2.imwrite(temp_path, frame)
            
            # 使用Canvas来显示图片，这样可以绘制ROI框
            from PIL import ImageTk
            canvas_frame = tk.Frame(left_frame)
            canvas_frame.pack()
            
            canvas = tk.Canvas(canvas_frame, bg="gray")
            canvas.pack(padx=10, pady=10)
            
            # 缩放比例
            display_width = 800
            display_height = int(height * display_width / width)
            if display_height > 500:
                display_height = 500
                display_width = int(width * display_height / height)
            
            scale_x = display_width / width
            scale_y = display_height / height
            
            canvas.config(width=display_width, height=display_height)
            
            # 加载并显示图片
            pil_image = Image.open(temp_path)
            pil_image = pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.image = photo  # 保持引用
            
            # 右侧：ROI设置和预览
            right_frame = tk.Frame(main_frame, width=300)
            right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
            right_frame.pack_propagate(False)
            
            # ROI输入区域
            roi_input_label = tk.Label(right_frame, text="ROI区域设置", 
                                       font=("Arial", 12, "bold"))
            roi_input_label.pack(pady=10)
            
            # 创建输入框（使用与主窗口相同的变量）
            input_grid = tk.Frame(right_frame)
            input_grid.pack(pady=10)
            
            tk.Label(input_grid, text="X:", font=("Arial", 10)).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
            roi_x_entry = tk.Entry(input_grid, textvariable=self.roi_x_var, width=10, font=("Arial", 10))
            roi_x_entry.grid(row=0, column=1, padx=5, pady=5)
            
            tk.Label(input_grid, text="Y:", font=("Arial", 10)).grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
            roi_y_entry = tk.Entry(input_grid, textvariable=self.roi_y_var, width=10, font=("Arial", 10))
            roi_y_entry.grid(row=1, column=1, padx=5, pady=5)
            
            tk.Label(input_grid, text="宽度:", font=("Arial", 10)).grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
            roi_w_entry = tk.Entry(input_grid, textvariable=self.roi_w_var, width=10, font=("Arial", 10))
            roi_w_entry.grid(row=2, column=1, padx=5, pady=5)
            
            tk.Label(input_grid, text="高度:", font=("Arial", 10)).grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
            roi_h_entry = tk.Entry(input_grid, textvariable=self.roi_h_var, width=10, font=("Arial", 10))
            roi_h_entry.grid(row=3, column=1, padx=5, pady=5)
            
            # ROI预览区域
            roi_preview_label = tk.Label(right_frame, text="ROI区域预览", 
                                         font=("Arial", 12, "bold"))
            roi_preview_label.pack(pady=(20, 10))
            
            roi_preview_canvas = tk.Canvas(right_frame, width=280, height=200, bg="black")
            roi_preview_canvas.pack(pady=10)
            
            # OCR测试结果区域
            ocr_result_label = tk.Label(right_frame, text="OCR识别结果", 
                                        font=("Arial", 12, "bold"))
            ocr_result_label.pack(pady=(10, 5))
            
            ocr_result_text = tk.Text(right_frame, width=30, height=4,  # 减少高度
                                     font=("Consolas", 9), wrap=tk.WORD)
            ocr_result_text.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
            ocr_result_text.insert(tk.END, "点击'测试OCR'查看识别结果")
            ocr_result_text.config(state=tk.DISABLED)
            
            # 更新ROI预览的函数
            def update_roi_preview():
                """更新ROI框和预览"""
                try:
                    # 清除之前的ROI框
                    canvas.delete("roi_rect")
                    canvas.delete("roi_text")
                    
                    # 获取ROI值
                    x_str = self.roi_x_var.get()
                    y_str = self.roi_y_var.get()
                    w_str = self.roi_w_var.get()
                    h_str = self.roi_h_var.get()
                    
                    if x_str and y_str and w_str and h_str:
                        try:
                            x = int(x_str)
                            y = int(y_str)
                            w = int(w_str)
                            h = int(h_str)
                            
                            if w > 0 and h > 0 and x >= 0 and y >= 0:
                                # 转换为显示坐标
                                disp_x = x * scale_x
                                disp_y = y * scale_y
                                disp_w = w * scale_x
                                disp_h = h * scale_y
                                
                                # 确保在画布范围内
                                if disp_x < display_width and disp_y < display_height:
                                    # 绘制ROI矩形框（红色，3像素宽）
                                    canvas.create_rectangle(
                                        disp_x, disp_y, 
                                        disp_x + disp_w, disp_y + disp_h,
                                        outline="red", width=3, tags="roi_rect"
                                    )
                                    
                                    # 绘制标签
                                    canvas.create_text(
                                        disp_x + 5, disp_y + 5,
                                        text=f"ROI: {x},{y} {w}x{h}",
                                        fill="red", anchor=tk.NW,
                                        font=("Arial", 10, "bold"),
                                        tags="roi_text"
                                    )
                                    
                                    # 更新ROI预览（显示ROI区域的放大图）
                                    if y + h <= height and x + w <= width:
                                        roi_region = original_frame[y:y+h, x:x+w]
                                        if roi_region.size > 0:
                                            # 调整ROI预览大小
                                            roi_pil = Image.fromarray(cv2.cvtColor(roi_region, cv2.COLOR_BGR2RGB))
                                            preview_w, preview_h = 280, 200
                                            roi_pil = roi_pil.resize((preview_w, preview_h), Image.Resampling.LANCZOS)
                                            roi_photo = ImageTk.PhotoImage(roi_pil)
                                            roi_preview_canvas.delete("all")
                                            roi_preview_canvas.create_image(0, 0, anchor=tk.NW, image=roi_photo)
                                            roi_preview_canvas.image = roi_photo
                                    
                        except ValueError:
                            pass
                    else:
                        # 清除ROI预览
                        roi_preview_canvas.delete("all")
                        roi_preview_canvas.create_text(140, 100, text="请输入ROI参数", 
                                                      fill="white", font=("Arial", 12))
                        
                except Exception as e:
                    print(f"更新ROI预览错误: {e}")
            
            # 绑定输入框变化事件
            for entry in [roi_x_entry, roi_y_entry, roi_w_entry, roi_h_entry]:
                entry.bind("<KeyRelease>", lambda e: update_roi_preview())
                entry.bind("<FocusOut>", lambda e: update_roi_preview())
            
            # 测试OCR的函数
            def test_ocr():
                """测试当前ROI区域的OCR识别"""
                try:
                    x_str = self.roi_x_var.get()
                    y_str = self.roi_y_var.get()
                    w_str = self.roi_w_var.get()
                    h_str = self.roi_h_var.get()
                    
                    if not (x_str and y_str and w_str and h_str):
                        ocr_result_text.config(state=tk.NORMAL)
                        ocr_result_text.delete(1.0, tk.END)
                        ocr_result_text.insert(tk.END, "请先设置ROI区域")
                        ocr_result_text.config(state=tk.DISABLED)
                        return
                    
                    x = int(x_str)
                    y = int(y_str)
                    w = int(w_str)
                    h = int(h_str)
                    
                    if w <= 0 or h <= 0:
                        ocr_result_text.config(state=tk.NORMAL)
                        ocr_result_text.delete(1.0, tk.END)
                        ocr_result_text.insert(tk.END, "ROI尺寸无效")
                        ocr_result_text.config(state=tk.DISABLED)
                        return
                    
                    # 提取ROI区域
                    roi_frame = original_frame[y:y+h, x:x+w]
                    
                    # 使用与detect_replay_text相同的预处理方法（3个方法）
                    processed_images = self.cutter.preprocess_image(roi_frame)
                    
                    ocr_result_text.config(state=tk.NORMAL)
                    ocr_result_text.delete(1.0, tk.END)
                    ocr_result_text.insert(tk.END, "OCR识别结果：\n")
                    ocr_result_text.insert(tk.END, "=" * 30 + "\n\n")
                    
                    psm_modes = [6, 7, 11]  # 有效的PSM模式
                    found_replay = False
                    
                    # Pipeline方式测试：方法3 -> 方法7 -> 方法10
                    method_names = ["方法3: OTSU二值化", "方法7: 反转OTSU", "方法10: 锐化"]
                    
                    for img_idx, processed_img in enumerate(processed_images):
                        if img_idx < len(method_names):
                            ocr_result_text.insert(tk.END, f"{method_names[img_idx]}:\n")
                            pil_image = Image.fromarray(processed_img)
                            
                            for psm in psm_modes:
                                try:
                                    # 包含大小写字母的白名单
                                    config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=REPLAYreplay0123456789'
                                    text = pytesseract.image_to_string(pil_image, config=config)
                                    text_clean = text.strip().replace('\n', ' ')
                                    
                                    if text_clean:
                                        ocr_result_text.insert(tk.END, f"  PSM {psm}: '{text_clean}'\n")
                                        # 检查各种大小写组合
                                        text_upper = text_clean.upper()
                                        text_lower = text_clean.lower()
                                        if 'REPLAY' in text_upper or 'replay' in text_lower or 'Replay' in text_clean:
                                            ocr_result_text.insert(tk.END, "    ✓ 检测到REPLAY!\n")
                                            found_replay = True
                                        # 部分匹配
                                        elif 'repl' in text_lower or 'REPL' in text_upper:
                                            ocr_result_text.insert(tk.END, "    ⚠ 检测到部分匹配 (repl)\n")
                                            found_replay = True
                                except:
                                    pass
                            ocr_result_text.insert(tk.END, "\n")
                    
                    if found_replay:
                        ocr_result_text.insert(tk.END, "\n✓ 成功识别到REPLAY文字！\n")
                    else:
                        ocr_result_text.insert(tk.END, "\n✗ 未识别到REPLAY\n")
                        ocr_result_text.insert(tk.END, "建议：\n")
                        ocr_result_text.insert(tk.END, "1. 调整ROI区域\n")
                        ocr_result_text.insert(tk.END, "2. 确保ROI包含完整文字\n")
                        ocr_result_text.insert(tk.END, "3. 检查文字是否清晰可见")
                    
                    ocr_result_text.config(state=tk.DISABLED)
                    
                except Exception as e:
                    ocr_result_text.config(state=tk.NORMAL)
                    ocr_result_text.delete(1.0, tk.END)
                    ocr_result_text.insert(tk.END, f"测试失败: {e}")
                    ocr_result_text.config(state=tk.DISABLED)
            
            # 按钮区域（固定在底部）
            btn_frame = tk.Frame(right_frame)
            btn_frame.pack(side=tk.BOTTOM, pady=10, fill=tk.X)
            
            test_ocr_btn = tk.Button(btn_frame, text="测试OCR", 
                                    command=test_ocr,
                                    font=("Arial", 10), bg="lightgreen", width=15)
            test_ocr_btn.pack(pady=3, fill=tk.X)
            
            apply_btn = tk.Button(btn_frame, text="应用ROI", 
                                 command=lambda: [self.apply_roi(), update_roi_preview()],
                                 font=("Arial", 10), bg="lightblue", width=15)
            apply_btn.pack(pady=3, fill=tk.X)
            
            clear_btn = tk.Button(btn_frame, text="清除ROI", 
                                 command=lambda: [self.clear_roi(), update_roi_preview()],
                                 font=("Arial", 10), width=15)
            clear_btn.pack(pady=3, fill=tk.X)
            
            # 关闭按钮
            close_btn = tk.Button(btn_frame, text="关闭预览", 
                                 command=preview_window.destroy, 
                                 font=("Arial", 12), width=15)
            close_btn.pack(pady=5, fill=tk.X)
            
            # 初始更新（如果有已设置的ROI）
            if self.cutter.roi:
                x, y, w, h = self.cutter.roi
                self.roi_x_var.set(str(x))
                self.roi_y_var.set(str(y))
                self.roi_w_var.set(str(w))
                self.roi_h_var.set(str(h))
            
            # 初始更新预览
            update_roi_preview()
            
            # 窗口关闭时删除临时文件
            def on_close():
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                preview_window.destroy()
            
            preview_window.protocol("WM_DELETE_WINDOW", on_close)
            
        except Exception as e:
            messagebox.showerror("错误", f"预览失败: {e}")
    
    def apply_roi(self):
        """应用ROI设置"""
        try:
            x = int(self.roi_x_var.get()) if self.roi_x_var.get() else 0
            y = int(self.roi_y_var.get()) if self.roi_y_var.get() else 0
            w = int(self.roi_w_var.get()) if self.roi_w_var.get() else 0
            h = int(self.roi_h_var.get()) if self.roi_h_var.get() else 0
            
            if w <= 0 or h <= 0:
                messagebox.showerror("错误", "宽度和高度必须大于0")
                return
            
            if x < 0 or y < 0:
                messagebox.showerror("错误", "X和Y坐标不能为负数")
                return
            
            self.cutter.set_roi(x, y, w, h)
            self.roi_status_var.set(f"当前ROI: X={x}, Y={y}, 宽度={w}, 高度={h}")
            messagebox.showinfo("成功", f"ROI区域已设置:\nX={x}, Y={y}\n宽度={w}, 高度={h}")
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")
        except Exception as e:
            messagebox.showerror("错误", f"设置ROI失败: {e}")
    
    def clear_roi(self):
        """清除ROI设置"""
        self.cutter.clear_roi()
        self.roi_x_var.set("")
        self.roi_y_var.set("")
        self.roi_w_var.set("")
        self.roi_h_var.set("")
        self.roi_status_var.set("当前: 使用全帧检测")
        messagebox.showinfo("成功", "已清除ROI设置，将使用全帧检测")
    
    def update_progress(self, value: float, message: str):
        """更新进度"""
        self.progress_bar['value'] = value
        self.progress_var.set(message)
        self.root.update_idletasks()
    
    def analyze_video(self):
        """分析视频"""
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showerror("错误", "请先选择输入视频文件")
            return
        
        try:
            check_interval = float(self.interval_var.get())
            if check_interval <= 0:
                messagebox.showerror("错误", "检测间隔必须大于0")
                return
            self.cutter.check_interval = check_interval
        except ValueError:
            messagebox.showerror("错误", "检测间隔必须是数字")
            return
        
        # 设置检测模式
        self.cutter.detection_mode = self.detection_mode_var.get()
        
        # 重置时间（开始新的分析）
        self.analysis_time = 0.0
        self.cut_time = 0.0
        self.time_info_var.set("总操作时长: 0.0秒")
        
        # 禁用按钮
        self.analyze_btn.config(state=tk.DISABLED)
        self.cut_btn.config(state=tk.DISABLED)
        
        def analyze_thread():
            try:
                # 记录分析开始时间
                analysis_start_time = time.time()
                
                mode_name = "快速模式" if self.cutter.detection_mode == "fast" else "全面模式"
                self.update_progress(0, f"开始分析视频... ({mode_name})")
                segments = self.cutter.analyze_video(self.video_path, self.update_progress)
                self.segments = segments
                
                # 记录分析结束时间并计算耗时
                analysis_end_time = time.time()
                self.analysis_time = analysis_end_time - analysis_start_time
                
                # 更新片段信息
                self.segments_text.delete(1.0, tk.END)
                if segments:
                    total_time = sum(end - start for start, end in segments)
                    self.segments_text.insert(tk.END, f"找到 {len(segments)} 个片段，总时长: {total_time:.2f} 秒\n\n")
                    for i, (start, end) in enumerate(segments, 1):
                        self.segments_text.insert(tk.END, f"片段 {i}: {start:.2f}秒 - {end:.2f}秒 (时长: {end-start:.2f}秒)\n")
                    self.cut_btn.config(state=tk.NORMAL)
                else:
                    self.segments_text.insert(tk.END, "未找到可保留的片段（所有片段都包含REPLAY字样）")
                
                # 更新总时长显示
                total_operation_time = self.analysis_time + self.cut_time
                self.time_info_var.set(f"总操作时长: {total_operation_time:.2f}秒 (分析: {self.analysis_time:.2f}秒, 剪辑: {self.cut_time:.2f}秒)")
                
                self.update_progress(100, f"分析完成！找到 {len(segments)} 个片段，耗时: {self.analysis_time:.2f}秒")
                self.analyze_btn.config(state=tk.NORMAL)
                
            except Exception as e:
                messagebox.showerror("错误", f"分析失败: {e}")
                self.update_progress(0, "分析失败")
                self.analyze_btn.config(state=tk.NORMAL)
        
        threading.Thread(target=analyze_thread, daemon=True).start()
    
    def cut_video(self):
        """剪辑视频"""
        if not self.output_folder:
            messagebox.showerror("错误", "请先选择输出文件夹")
            return
        
        if not self.segments:
            messagebox.showerror("错误", "没有可剪辑的片段，请先分析视频")
            return
        
        # 获取文件前缀
        file_prefix = self.output_prefix_var.get().strip()
        if not file_prefix:
            file_prefix = "race"
        
        # 重置剪辑时间（保留分析时间）
        self.cut_time = 0.0
        total_operation_time = self.analysis_time + self.cut_time
        self.time_info_var.set(f"总操作时长: {total_operation_time:.2f}秒 (分析: {self.analysis_time:.2f}秒, 剪辑: {self.cut_time:.2f}秒)")
        
        # 禁用按钮
        self.analyze_btn.config(state=tk.DISABLED)
        self.cut_btn.config(state=tk.DISABLED)
        
        def cut_thread():
            try:
                # 记录剪辑开始时间
                cut_start_time = time.time()
                
                output_files = self.cutter.cut_video(
                    self.video_path, 
                    self.segments, 
                    self.output_folder,
                    file_prefix,
                    self.update_progress
                )
                
                # 记录剪辑结束时间并计算耗时
                cut_end_time = time.time()
                self.cut_time = cut_end_time - cut_start_time
                
                # 更新总时长显示
                total_operation_time = self.analysis_time + self.cut_time
                self.time_info_var.set(f"总操作时长: {total_operation_time:.2f}秒 (分析: {self.analysis_time:.2f}秒, 剪辑: {self.cut_time:.2f}秒)")
                
                # 生成文件列表消息
                files_info = "\n".join([os.path.basename(f) for f in output_files])
                messagebox.showinfo(
                    "成功", 
                    f"视频剪辑完成！\n\n共生成 {len(output_files)} 个文件：\n{files_info}\n\n保存位置: {self.output_folder}\n\n总操作时长: {total_operation_time:.2f}秒 (分析: {self.analysis_time:.2f}秒, 剪辑: {self.cut_time:.2f}秒)"
                )
                self.update_progress(0, f"剪辑完成，耗时: {self.cut_time:.2f}秒")
                self.analyze_btn.config(state=tk.NORMAL)
                self.cut_btn.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("错误", f"剪辑失败: {e}")
                self.update_progress(0, "剪辑失败")
                self.analyze_btn.config(state=tk.NORMAL)
                self.cut_btn.config(state=tk.NORMAL)
        
        threading.Thread(target=cut_thread, daemon=True).start()
    
    def run(self):
        """运行GUI"""
        self.root.mainloop()


def main():
    """主函数"""
    try:
        app = VideoCutterGui()
        app.run()
    except KeyboardInterrupt:
        print("\n程序已退出")
    except Exception as e:
        print(f"程序错误: {e}")


if __name__ == "__main__":
    main()
