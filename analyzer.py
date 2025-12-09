from PIL import ImageFont, ImageDraw, Image
import cv2
import mediapipe as mp
import numpy as np
import os
import subprocess
import pandas as pd

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def calculate_torso_angle(shoulder_mid, hip_mid):
    vector = np.array([hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1]])
    vertical = np.array([0, 1])
    cosine = np.dot(vector, vertical) / (np.linalg.norm(vector) * np.linalg.norm(vertical))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def calculate_inclination_angle(center, foot_mid):
    dx, dy = foot_mid[0] - center[0], foot_mid[1] - center[1]
    if dy == 0 and dx == 0:
        return 0.0
    if dy == 0:
        return np.nan
    return np.degrees(np.arctan(abs(dx) / abs(dy)))
    
def calculate_ski_tilt_signed(ankle, toe):
    dx, dy = toe[0] - ankle[0], toe[1] - ankle[1]
    if dx == 0 and dy == 0:
        return np.nan
    
    ski_vec = np.array([dx, dy])
    vertical = np.array([0, -1])  # 垂直基準（画面上方向）

    det = vertical[0]*ski_vec[1] - vertical[1]*ski_vec[0]  # 外積
    dot = vertical[0]*ski_vec[0] + vertical[1]*ski_vec[1]  # 内積

    angle = np.degrees(np.arctan2(det, dot))  # [-180°, 180°]
    return angle

def smooth(history, window=5):
    if len(history) < window:
        return np.mean(history)
    return np.mean(history[-window:])
    
def resize_keep_aspect(img, target_width=None, target_height=None):
    h, w = img.shape[:2]
    if target_width and not target_height:
        scale = target_width / w
    elif target_height and not target_width:
        scale = target_height / h
    else:
        raise ValueError("どちらか一方だけ指定してください")
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h))

def merge_audio(original_path, processed_path):
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original video file not found: {original_path}")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed video file not found: {processed_path}")

    output_path = os.path.splitext(processed_path)[0] + "_with_audio.mp4"

    # ffprobeで音声ストリームの有無を確認
    probe = subprocess.run(
        ['ffprobe', '-i', original_path, '-show_streams', '-select_streams', 'a', '-loglevel', 'error'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    has_audio = bool(probe.stdout.decode().strip())

    if has_audio:
        # 音声あり → マージ
        command = [
            'ffmpeg', '-y',
            '-i', original_path,
            '-i', processed_path,
            '-c:v', 'copy',   # 映像は再エンコードせずコピー
            '-c:a', 'aac',
            '-map', '0:a?',
            '-map', '1:v?',
            output_path
        ]
    else:
        # 音声なし → 映像のみ
        command = [
            'ffmpeg', '-y',
            '-i', processed_path,
            '-c:v', 'copy',
            output_path
        ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("FFmpeg stderr:", result.stderr.decode())
    if not os.path.exists(output_path):
        print("FFmpeg stderr:", result.stderr.decode())
        raise FileNotFoundError(f"Failed to create output file: {output_path}")

    return output_path
    
def safe(val):
    return "--" if np.isnan(val) else f"{int(val)}"
    
def process_video(input_path, progress_callback=None, show_background=True, selected_angles=None):
    left_knee_history = []
    right_knee_history = []
    inclination_history = []

    mp_pose = mp.solutions.pose
    KEYPOINTS = {
        "nose": 0,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28,
        "left_foot_index": 31, "right_foot_index": 32
    }

    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("動画を読み込めませんでした")

    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_output_path = os.path.splitext(input_path)[0] + "_processed_temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (720, 1280))  

    # 事前に全フェーズ画像の横幅を調べて最大値を決定
    phase_paths = [
        "image/turn_phase_neutral.png",
        "image/turn_phase_left.png",
        "image/turn_phase_right.png",
        "image/turn_phase_none.png"
    ]
    
    widths = []
    for path in phase_paths:
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                widths.append(w)
        
    max_width = max(widths) if widths else 1   # 最大横幅
    target_width = width // 4                  # 動画幅の半分
    scale = target_width / max_width           # 縮尺率
    with mp_pose.Pose() as pose:
        while ret:
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if progress_callback:
                progress_callback(current_frame / total_frames)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            image = frame.copy() 
            canvas = np.zeros((1280, 720, 3), dtype=np.uint8)

            grid_data = []
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                joints = {}
                for name, idx in KEYPOINTS.items():
                    if idx < len(lm):
                        x = int(lm[idx].x * width)
                        y = int(lm[idx].y * height)
                        joints[name] = (x, y)
                  

                required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_ankle", "right_ankle"]
                if all(k in joints for k in required):
                    shoulder_mid = ((joints["left_shoulder"][0] + joints["right_shoulder"][0]) // 2,
                                    (joints["left_shoulder"][1] + joints["right_shoulder"][1]) // 2)
                    hip_mid = ((joints["left_hip"][0] + joints["right_hip"][0]) // 2,
                               (joints["left_hip"][1] + joints["right_hip"][1]) // 2)
                    foot_mid = ((joints["left_ankle"][0] + joints["right_ankle"][0]) // 2,
                                (joints["left_ankle"][1] + joints["right_ankle"][1]) // 2)
                    center = ((shoulder_mid[0] + hip_mid[0]) // 2, (shoulder_mid[1] + hip_mid[1]) // 2)

                    torso_angle = calculate_torso_angle(shoulder_mid, hip_mid)
                    inclination_angle = calculate_inclination_angle(center, foot_mid)

                    left_knee_angle = calculate_angle(joints["left_hip"], joints["left_knee"], joints["left_ankle"])
                    right_knee_angle = calculate_angle(joints["right_hip"], joints["right_knee"], joints["right_ankle"])
                    left_hip_angle = calculate_angle(joints["left_knee"], joints["left_hip"], joints["right_hip"])
                    right_hip_angle = calculate_angle(joints["right_knee"], joints["right_hip"], joints["left_hip"])
                    left_abduction_angle = calculate_angle(hip_mid, joints["left_hip"], joints["left_knee"])
                    right_abduction_angle = calculate_angle(hip_mid, joints["right_hip"], joints["right_knee"])
                    left_knee_abduction = calculate_angle(hip_mid, joints["left_knee"], joints["left_ankle"])
                    right_knee_abduction = calculate_angle(hip_mid, joints["right_knee"], joints["right_ankle"])
                    ski_tilt_angle = calculate_ski_tilt_signed(joints["left_ankle"], joints["left_foot_index"])

                    
                    # 履歴に追加
                    left_knee_history.append(left_knee_angle)
                    right_knee_history.append(right_knee_angle)
                    inclination_history.append(inclination_angle)
                    
                    # 平滑化
                    left_knee_s = smooth(left_knee_history)
                    right_knee_s = smooth(right_knee_history)
                    inclination_s = smooth(inclination_history)
                    
                    inclination_display = "--" if np.isnan(inclination_s) else f"{inclination_s:.1f}"
                    
                    if np.isnan(inclination_s):
                        turn_phase = "--"
                    elif inclination_s <= 11.0:
                        turn_phase = "Neutral"
                    else:
                        left_knee_sum = left_knee_abduction + left_knee_s
                        right_knee_sum = right_knee_abduction + right_knee_s
                        if left_knee_sum > right_knee_sum:
                            primary = "Left"
                            
                        else:
                            primary = "Right"
                            
                        turn_phase = f"{primary}"
                                            
                    if turn_phase == "Neutral":
                        phase_img_path = "image/turn_phase_neutral.png"
                    elif turn_phase == "Left":
                        phase_img_path = "image/turn_phase_left.png"
                    elif turn_phase == "Right":
                        phase_img_path = "image/turn_phase_right.png"
                    else:
                        phase_img_path = "image/turn_phase_none.png"
                        

                    # 骨格ラインと関節点は image に描画
                    connections = [
                        ("left_ankle", "left_knee"), ("left_knee", "left_hip"),
                        ("right_ankle", "right_knee"), ("right_knee", "right_hip"),
                        ("left_hip", "right_hip"), ("left_shoulder", "right_shoulder"),
                        ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
                        ("right_shoulder", "right_hip"), ("left_shoulder", "left_hip"),
                        ("right_elbow", "right_wrist"), ("left_elbow", "left_wrist")
                    ]
                    for a, b in connections:
                        if a in joints and b in joints:
                            pt1, pt2 = joints[a], joints[b]
                            color = (255, 0, 255)
                            cv2.line(image, pt1, pt2, color, 2)

                    for name, (x, y) in joints.items():
                        cv2.circle(image, (x, y), 2, (255, 0, 255), -1)
                    
                    # 元動画を縦型にリサイズ
                    video_resized = resize_keep_aspect(image, target_width=720)
                    h, w = video_resized.shape[:2]
                    

                    
                    # 中央に配置したい場合
                    x_offset = (canvas.shape[1] - w) // 2
                    y_offset = 0
                    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = video_resized
                        
                                      
                    grid_data = [
                        ["L-Knee Ext/Flex", safe(left_knee_angle)],
                        ["R-Knee Ext/Flex", safe(right_knee_angle)],
                        ["L-Knee Abd/Add", safe(left_knee_abduction)],
                        ["R-Knee Abd/Add", safe(right_knee_abduction)],
                        ["L-Hip Ext/Flex", safe(left_hip_angle)],
                        ["R-Hip Ext/Flex", safe(right_hip_angle)],
                        ["L-Hip Abd/Add", safe(left_abduction_angle)],
                        ["R-Hip Abd/Add", safe(right_abduction_angle)],
                        ["Torso Tilt", f"{torso_angle:.1f}"],
                        ["Inclination Angle", inclination_display]
                    ]
                    cell_height = 40
                    start_x = 30
                    start_y = canvas.shape[0] - len(grid_data)*cell_height - 30
                    img_pil = Image.fromarray(canvas)
                    draw = ImageDraw.Draw(img_pil)
                    font_path="static/BestTen-CRT.otf"
                    font = ImageFont.truetype(font_path, 20)
                                                         
                    for i, (label, value) in enumerate(grid_data):
                        y_pos = start_y + i * cell_height
                        top_left = (start_x, start_y + i * 40)
                        bottom_right = (start_x + 300, start_y + (i + 1) * 40)     
                        draw.rectangle([top_left, bottom_right], fill=(0,0,0), outline=(255,255,255))
                        draw.text((35, y_pos+10), label, font=font, fill=(255,255,255))
                        draw.text((200, y_pos+10), value, font=font, fill=(255,255,255))
                        
                    canvas = np.array(img_pil)
                    
                    box_width, box_height = 300, 100

                    
                    if phase_img_path and os.path.exists(phase_img_path):
                        phase_img = cv2.imread(phase_img_path)
                        if phase_img is not None:
                            # 最大横幅に合わせた縮尺率でリサイズ
                            h, w = phase_img.shape[:2]
                            new_w, new_h = int(w * scale*1.25), int(h * scale*1.25)
                            phase_resized = cv2.resize(phase_img, (new_w, new_h))
                            
              
                            # 貼り付け位置（中央寄せ）
                            h, w = phase_resized.shape[:2]
                            x_offset = 30
                            y_offset = canvas.shape[0] // 2 + 55
                            canvas[y_offset:y_offset+h, x_offset:x_offset+w] = phase_resized
                        
                        turn_phase_path = "image/turn_phase.png"
                        turn_phase = cv2.imread(turn_phase_path)
                        h, w = turn_phase.shape[:2]
                        new_w, new_h = int(w * scale*1.25), int(h * scale*1.25)
                        turn_phase_resized = cv2.resize(turn_phase, (new_w, new_h))
                        h, w = turn_phase_resized.shape[:2]
                        x_offset = 30
                        y_offset = canvas.shape[0] // 2
                        canvas [y_offset:y_offset+h, x_offset:x_offset+w] = turn_phase_resized
            else:
                # 骨格未検出時は元フレームをそのまま表示
                video_resized = resize_keep_aspect(image, target_width=720)
                h, w = video_resized.shape[:2]
                x_offset = (canvas.shape[1] - w) // 2
                y_offset = 0
                canvas[y_offset:y_offset+h, x_offset:x_offset+w] = video_resized
            
                # グリッドは "--" を表示
                grid_data = [
                    ["L-Knee Ext/Flex", "--"],
                    ["R-Knee Ext/Flex", "--"],
                    ["L-Knee Abd/Add", "--"],
                    ["R-Knee Abd/Add", "--"],
                    ["L-Hip Ext/Flex", "--"],
                    ["R-Hip Ext/Flex", "--"],
                    ["L-Hip Abd/Add", "--"],
                    ["R-Hip Abd/Add", "--"],
                    ["Torso Tilt", "--"],
                    ["Inclination Angle", "--"]
                ] 
                cell_height = 40
                start_x = 30
                start_y = canvas.shape[0] - len(grid_data)*cell_height - 30
                img_pil = Image.fromarray(canvas)
                draw = ImageDraw.Draw(img_pil)
                font_path="static/BestTen-CRT.otf"
                font = ImageFont.truetype(font_path, 20)
                            
                for i, (label, value) in enumerate(grid_data):
                    y_pos = start_y + i * cell_height
                    top_left = (start_x, start_y + i * 40)
                    bottom_right = (start_x + 300, start_y + (i + 1) * 40)  
                    draw.rectangle([top_left, bottom_right], fill=(0,0,0), outline=(255,255,255))
                    draw.text((35, y_pos+10), label, font=font, fill=(255,255,255))
                    draw.text((200, y_pos+10), value, font=font, fill=(255,255,255))
                canvas = np.array(img_pil)
                
                box_width, box_height = 300, 100
                        

                                                           
           # 書き出し
            out.write(canvas)
            ret, frame = cap.read()

    cap.release()
    out.release()

    final_output = merge_audio(input_path, temp_output_path)
    return final_output
