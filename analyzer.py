from PIL import ImageFont, ImageDraw, Image
import cv2
import mediapipe as mp
import numpy as np
import os
import subprocess
import pandas as pd
from scipy.spatial.transform import Rotation as R

# ==========================================
# 改善案から追加された3Dリギング用関数群
# ==========================================

def normalize_3d(joints_3d):
    """身長（鼻から股関節の距離）を基準にスケールを1.7m相当に正規化する"""
    if "left_hip" not in joints_3d or "nose" not in joints_3d:
        return joints_3d
    hip = joints_3d["left_hip"]
    head = joints_3d["nose"]
    height = np.linalg.norm(head - hip)

    [cite_start]scale = 1.7 / (height + 1e-6) [cite: 4]
    for k in joints_3d:
        [cite_start]joints_3d[k] = joints_3d[k] * scale [cite: 4]
    return joints_3d

def compute_rotation(p, c, forward=np.array([0, 1, 0])):
    """2つの3D点から方向ベクトルを求め、基準方向からの回転(Rotationオブジェクト)を計算する"""
    [cite_start]direction = c - p [cite: 2]
    [cite_start]norm = np.linalg.norm(direction) [cite: 2]
    [cite_start]if norm < 1e-6: [cite: 2]
        [cite_start]return R.identity() [cite: 2]

    [cite_start]direction /= norm [cite: 2]
    [cite_start]axis = np.cross(forward, direction) [cite: 2]
    [cite_start]angle = np.arccos(np.clip(np.dot(forward, direction), -1.0, 1.0)) [cite: 2]

    [cite_start]if np.linalg.norm(axis) < 1e-6: [cite: 2]
        [cite_start]return R.identity() [cite: 2]

    [cite_start]axis /= np.linalg.norm(axis) [cite: 2]
    [cite_start]return R.from_rotvec(axis * angle) [cite: 2]

def compute_skeleton_rotations(joints_3d):
    """各関節の3D座標から、主要ボーンの回転を一括計算する"""
    rotations = {}

    # [cite_start]腕 [cite: 3]
    [cite_start]rotations["RightUpperArm"] = compute_rotation(joints_3d["right_shoulder"], joints_3d["right_elbow"]) [cite: 3]
    [cite_start]rotations["RightLowerArm"] = compute_rotation(joints_3d["right_elbow"], joints_3d["right_wrist"]) [cite: 3]
    [cite_start]rotations["LeftUpperArm"] = compute_rotation(joints_3d["left_shoulder"], joints_3d["left_elbow"]) [cite: 3]
    [cite_start]rotations["LeftLowerArm"] = compute_rotation(joints_3d["left_elbow"], joints_3d["left_wrist"]) [cite: 3]

    # [cite_start]脚 [cite: 3]
    [cite_start]rotations["RightUpperLeg"] = compute_rotation(joints_3d["right_hip"], joints_3d["right_knee"]) [cite: 3]
    [cite_start]rotations["RightLowerLeg"] = compute_rotation(joints_3d["right_knee"], joints_3d["right_ankle"]) [cite: 3]
    [cite_start]rotations["LeftUpperLeg"] = compute_rotation(joints_3d["left_hip"], joints_3d["left_knee"]) [cite: 3]
    [cite_start]rotations["LeftLowerLeg"] = compute_rotation(joints_3d["left_knee"], joints_3d["left_ankle"]) [cite: 3]

    return rotations

def smooth_rotations(prev_rotations, current_rotations, t=0.2):
    """QuaternionのSlerp（球面線形補間）を用いて回転のガタつきを滑らかにする"""
    if not prev_rotations:
        return current_rotations
    
    smoothed = {}
    for bone, curr_rot in current_rotations.items():
        if bone in prev_rotations:
            # scipyのSlerpライブラリに準拠した補間処理
            from scipy.spatial.transform import Slerp
            key_times = [0, 1]
            key_rots = R.from_quat([prev_rotations[bone].as_quat(), curr_rot.as_quat()])
            slerp = Slerp(key_times, key_rots)
            smoothed[bone] = slerp([t])[0]  # 前フレームの姿勢にtの割合で近づける
        else:
            smoothed[bone] = curr_rot
    return smoothed

# ==========================================
# 既存の解析用関数群（変更なし）
# ==========================================

def calculate_angle(a, b, c):
    [cite_start]a, b, c = np.array(a), np.array(b), np.array(c) [cite: 6]
    [cite_start]ba, bc = a - b, c - b [cite: 6]
    [cite_start]cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)) [cite: 6]
    [cite_start]return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) [cite: 6]

def calculate_torso_angle(shoulder_mid, hip_mid):
    [cite_start]vector = np.array([hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1]]) [cite: 6]
    [cite_start]vertical = np.array([0, 1]) [cite: 6]
    [cite_start]cosine = np.dot(vector, vertical) / (np.linalg.norm(vector) * np.linalg.norm(vertical)) [cite: 6]
    [cite_start]return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) [cite: 6]

def calculate_inclination_angle(center, foot_mid):
    [cite_start]dx, dy = foot_mid[0] - center[0], foot_mid[1] - center[1] [cite: 6, 7]
    [cite_start]if dy == 0 and dx == 0: [cite: 7]
        [cite_start]return 0.0 [cite: 7]
    [cite_start]if dy == 0: [cite: 7]
        [cite_start]return np.nan [cite: 7]
    [cite_start]return np.degrees(np.arctan(abs(dx) / abs(dy))) [cite: 7]
    
def calculate_ski_tilt_signed(ankle, toe):
    [cite_start]dx, dy = toe[0] - ankle[0], toe[1] - ankle[1] [cite: 7]
    [cite_start]if dx == 0 and dy == 0: [cite: 7]
        [cite_start]return np.nan [cite: 7]
    
    [cite_start]ski_vec = np.array([dx, dy]) [cite: 8]
    [cite_start]vertical = np.array([0, -1])  # 垂直基準（画面上方向） [cite: 8]

    [cite_start]det = vertical[0]*ski_vec[1] - vertical[1]*ski_vec[0]  # 外積 [cite: 8]
    [cite_start]dot = vertical[0]*ski_vec[0] + vertical[1]*ski_vec[1]  # 内積 [cite: 8]

    [cite_start]angle = np.degrees(np.arctan2(det, dot))  # [-180°, 180°] [cite: 8]
    return angle

def smooth(history, window=5):
    [cite_start]if len(history) < window: [cite: 8]
        [cite_start]return np.mean(history) [cite: 8]
    [cite_start]return np.mean(history[-window:]) [cite: 8]
    
def resize_keep_aspect(img, target_width=None, target_height=None):
    [cite_start]h, w = img.shape[:2] [cite: 8]
    [cite_start]if target_width and not target_height: [cite: 8]
        [cite_start]scale = target_width / w [cite: 9]
    [cite_start]elif target_height and not target_width: [cite: 9]
        [cite_start]scale = target_height / h [cite: 9]
    else:
        [cite_start]raise ValueError("どちらか一方だけ指定してください") [cite: 9]
    [cite_start]new_w, new_h = int(w * scale), int(h * scale) [cite: 9]
    [cite_start]return cv2.resize(img, (new_w, new_h)) [cite: 9]

def merge_audio(original_path, processed_path):
    [cite_start]if not os.path.exists(original_path): [cite: 9]
        [cite_start]raise FileNotFoundError(f"Original video file not found: {original_path}") [cite: 9]
    [cite_start]if not os.path.exists(processed_path): [cite: 9]
        [cite_start]raise FileNotFoundError(f"Processed video file not found: {processed_path}") [cite: 9, 10]

    [cite_start]output_path = os.path.splitext(processed_path)[0] + "_with_audio.mp4" [cite: 10]

    [cite_start]probe = subprocess.run( [cite: 10]
        [cite_start]['ffprobe', '-i', original_path, '-show_streams', '-select_streams', 'a', '-loglevel', 'error'], [cite: 10]
        [cite_start]stdout=subprocess.PIPE, stderr=subprocess.PIPE [cite: 10]
    )
    [cite_start]has_audio = bool(probe.stdout.decode().strip()) [cite: 10]

    [cite_start]if has_audio: [cite: 10]
        [cite_start]command = [ [cite: 10]
            [cite_start]'ffmpeg', '-y', [cite: 10]
            [cite_start]'-i', original_path, [cite: 11]
            [cite_start]'-i', processed_path, [cite: 11]
            [cite_start]'-c:v', 'copy', [cite: 11]
            [cite_start]'-c:a', 'aac', [cite: 11]
            [cite_start]'-map', '0:a?', [cite: 11]
            [cite_start]'-map', '1:v?', [cite: 11]
            [cite_start]output_path [cite: 11]
        ]
    [cite_start]else: [cite: 12]
        [cite_start]command = [ [cite: 12]
            [cite_start]'ffmpeg', '-y', [cite: 12]
            [cite_start]'-i', processed_path, [cite: 12]
            [cite_start]'-c:v', 'copy', [cite: 12]
            [cite_start]output_path [cite: 12]
        ]

    [cite_start]result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) [cite: 12]
    [cite_start]print("FFmpeg stderr:", result.stderr.decode()) [cite: 12]
    [cite_start]if not os.path.exists(output_path): [cite: 12, 13]
        [cite_start]print("FFmpeg stderr:", result.stderr.decode()) [cite: 13]
        [cite_start]raise FileNotFoundError(f"Failed to create output file: {output_path}") [cite: 13]

    [cite_start]return output_path [cite: 13]
    
def safe(val):
    [cite_start]return "--" if np.isnan(val) else f"{int(val)}" [cite: 13]
    
def process_video(input_path, progress_callback=None, show_background=True, selected_angles=None):
    [cite_start]left_knee_history = [] [cite: 13]
    [cite_start]right_knee_history = [] [cite: 13]
    [cite_start]inclination_history = [] [cite: 13]
    
    # クォータニオン平滑化用の前フレーム保持変数
    prev_rotations = {}

    [cite_start]mp_pose = mp.solutions.pose [cite: 13]
    KEYPOINTS = {
        [cite_start]"nose": 0, [cite: 13]
        [cite_start]"left_shoulder": 11, "right_shoulder": 12, [cite: 14]
        [cite_start]"left_elbow": 13, "right_elbow": 14, [cite: 14]
        [cite_start]"left_wrist": 15, "right_wrist": 16, [cite: 14]
        [cite_start]"left_hip": 23, "right_hip": 24, [cite: 14]
        [cite_start]"left_knee": 25, "right_knee": 26, [cite: 14]
        [cite_start]"left_ankle": 27, "right_ankle": 28, [cite: 14]
        [cite_start]"left_foot_index": 31, "right_foot_index": 32 [cite: 14]
    }

    [cite_start]cap = cv2.VideoCapture(input_path) [cite: 14]
    [cite_start]ret, frame = cap.read() [cite: 14]
    [cite_start]if not ret: [cite: 14]
        [cite_start]raise RuntimeError("動画を読み込めませんでした") [cite: 15]

    [cite_start]height, width = frame.shape[:2] [cite: 15]
    [cite_start]fps = cap.get(cv2.CAP_PROP_FPS) [cite: 15]
    [cite_start]total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) [cite: 15]

    [cite_start]temp_output_path = os.path.splitext(input_path)[0] + "_processed_temp.mp4" [cite: 15]
    [cite_start]fourcc = cv2.VideoWriter_fourcc(*'mp4v') [cite: 15]
    [cite_start]out = cv2.VideoWriter(temp_output_path, fourcc, fps, (720, 1280))   [cite: 15]

    [cite_start]phase_paths = [ [cite: 15]
        [cite_start]"image/turn_phase_neutral.png", [cite: 15]
        [cite_start]"image/turn_phase_left.png", [cite: 15]
        [cite_start]"image/turn_phase_right.png", [cite: 15]
        [cite_start]"image/turn_phase_none.png" [cite: 15]
    ]
    
    [cite_start]widths = [] [cite: 16]
    [cite_start]for path in phase_paths: [cite: 16]
        [cite_start]if os.path.exists(path): [cite: 16]
            [cite_start]img = cv2.imread(path) [cite: 16]
            [cite_start]if img is not None: [cite: 16]
                [cite_start]h, w = img.shape[:2] [cite: 16]
                [cite_start]widths.append(w) [cite: 16]
        
    [cite_start]max_width = max(widths) if widths else 1   # 最大横幅 [cite: 16, 17]
    [cite_start]target_width = width // 4                  # 動画幅の半分 [cite: 17]
    [cite_start]scale = target_width / max_width           # 縮尺率 [cite: 17]
    
    [cite_start]with mp_pose.Pose() as pose: [cite: 17]
        [cite_start]while ret: [cite: 17]
            [cite_start]current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) [cite: 17]
            [cite_start]if progress_callback: [cite: 17]
                [cite_start]progress_callback(current_frame / total_frames) [cite: 18]

            [cite_start]image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) [cite: 18]
            [cite_start]results = pose.process(image_rgb) [cite: 18]
            [cite_start]image = frame.copy()  [cite: 18]
            [cite_start]canvas = np.zeros((1280, 720, 3), dtype=np.uint8) [cite: 18]

            [cite_start]grid_data = [] [cite: 18]
            
            [cite_start]if results.pose_landmarks: [cite: 19]
                [cite_start]lm = results.pose_landmarks.landmark [cite: 19]
                
                # ----------------==================================
                # 1. 描画・既存解析用：2Dピクセル座標 (従来通り)
                # ----------------==================================
                [cite_start]joints_2d = {} [cite: 19]
                [cite_start]for name, idx in KEYPOINTS.items(): [cite: 19]
                    [cite_start]if idx < len(lm): [cite: 19]
                        [cite_start]x = int(lm[idx].x * width) [cite: 20]
                        [cite_start]y = int(lm[idx].y * height) [cite: 20]
                        [cite_start]joints_2d[name] = (x, y) [cite: 20]
                
                # ----------------==================================
                # 2. リギング用：疑似3D空間ベクトル座標 (改善案の統合)
                # ----------------==================================
                joints_3d = {}
                for name, idx in KEYPOINTS.items():
                    if idx < len(lm):
                        [cite_start]x_3d = lm[idx].x * width [cite: 5]
                        [cite_start]y_3d = -lm[idx].y * height  # 座標系の補正（Y軸反転） [cite: 1, 5]
                        [cite_start]z_3d = lm[idx].z * width    # Z深さ成分の適用 [cite: 5]
                        joints_3d[name] = np.array([x_3d, y_3d, z_3d])
                
                # 身長スケール正規化とボーン回転計算の適用
                [cite_start]joints_3d = normalize_3d(joints_3d) [cite: 5]
                [cite_start]current_rotations = compute_skeleton_rotations(joints_3d) [cite: 5]
                
                # クォータニオンのスムージング処理
                smoothed_rotations = smooth_rotations(prev_rotations, current_rotations, t=0.2)
                prev_rotations = smoothed_rotations  # 次フレームに引き継ぎ
                
                # [cite_start]【デバッグ/実用確認用】計算されたクォータニオンをコンソールへ出力 [cite: 5]
                # 本番運用の際は、ここでファイル出力や3D描画用バッファに格納してください
                # for bone, rot in smoothed_rotations.items():
                #     [cite_start]print(bone, rot.as_quat()) [cite: 5]

                # ----------------==================================
                # 3. 既存の描画・数値解析処理（joints を joints_2d に変更）
                # ----------------==================================
                [cite_start]required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_ankle", "right_ankle"] [cite: 20, 21]
                [cite_start]if all(k in joints_2d for k in required): [cite: 21]
                    [cite_start]shoulder_mid = ((joints_2d["left_shoulder"][0] + joints_2d["right_shoulder"][0]) // 2, [cite: 21]
                                    (joints_2d["left_shoulder"][1] + joints_2d["right_shoulder"][1]) [cite_start]// 2) [cite: 21]
                    [cite_start]hip_mid = ((joints_2d["left_hip"][0] + joints_2d["right_hip"][0]) // 2, [cite: 22]
                               (joints_2d["left_hip"][1] + joints_2d["right_hip"][1]) [cite_start]// 2) [cite: 22]
                    [cite_start]foot_mid = ((joints_2d["left_ankle"][0] + joints_2d["right_ankle"][0]) // 2, [cite: 22]
                                (joints_2d["left_ankle"][1] + joints_2d["right_ankle"][1]) [cite_start]// 2) [cite: 22, 23]
                    [cite_start]center = ((shoulder_mid[0] + hip_mid[0]) // 2, (shoulder_mid[1] + hip_mid[1]) // 2) [cite: 23]

                    [cite_start]torso_angle = calculate_torso_angle(shoulder_mid, hip_mid) [cite: 23]
                    [cite_start]inclination_angle = calculate_inclination_angle(center, foot_mid) [cite: 23]

                    [cite_start]left_knee_angle = calculate_angle(joints_2d["left_hip"], joints_2d["left_knee"], joints_2d["left_ankle"]) [cite: 24]
                    [cite_start]right_knee_angle = calculate_angle(joints_2d["right_hip"], joints_2d["right_knee"], joints_2d["right_ankle"]) [cite: 24]
                    [cite_start]left_hip_angle = calculate_angle(joints_2d["left_knee"], joints_2d["left_hip"], joints_2d["right_hip"]) [cite: 24]
                    [cite_start]right_hip_angle = calculate_angle(joints_2d["right_knee"], joints_2d["right_hip"], joints_2d["left_hip"]) [cite: 24, 25]
                    
                    [cite_start]left_abduction_angle = calculate_angle(hip_mid, joints_2d["left_hip"], joints_2d["left_knee"]) [cite: 25]
                    [cite_start]right_abduction_angle = calculate_angle(hip_mid, joints_2d["right_hip"], joints_2d["right_knee"]) [cite: 25]
                    [cite_start]left_knee_abduction = calculate_angle(hip_mid, joints_2d["left_knee"], joints_2d["left_ankle"]) [cite: 25]
                    [cite_start]right_knee_abduction = calculate_angle(hip_mid, joints_2d["right_knee"], joints_2d["right_ankle"]) [cite: 25, 26]
                    
                    [cite_start]ski_tilt_angle = calculate_ski_tilt_signed(joints_2d["left_ankle"], joints_2d["left_foot_index"]) [cite: 26]

                    [cite_start]left_knee_history.append(left_knee_angle) [cite: 26]
                    [cite_start]right_knee_history.append(right_knee_angle) [cite: 26]
                    [cite_start]inclination_history.append(inclination_angle) [cite: 26, 27]
                    
                    [cite_start]left_knee_s = smooth(left_knee_history) [cite: 27]
                    [cite_start]right_knee_s = smooth(right_knee_history) [cite: 27]
                    [cite_start]inclination_s = smooth(inclination_history) [cite: 27, 28]
                    
                    [cite_start]inclination_display = "--" if np.isnan(inclination_s) else f"{inclination_s:.1f}" [cite: 28]
                    
                    [cite_start]if np.isnan(inclination_s): [cite: 29]
                        [cite_start]turn_phase = "--" [cite: 29]
                    [cite_start]elif inclination_s <= 11.0: [cite: 29]
                        [cite_start]turn_phase = "Neutral" [cite: 29]
                    [cite_start]else: [cite: 29]
                        [cite_start]left_knee_sum = left_knee_abduction + left_knee_s [cite: 30]
                        [cite_start]right_knee_sum = right_knee_abduction + right_knee_s [cite: 30]
                        [cite_start]if left_knee_sum > right_knee_sum: [cite: 30]
                            [cite_start]primary = "Left" [cite: 31]
                        [cite_start]else: [cite: 31]
                            [cite_start]primary = "Right" [cite: 31]
                                            
                        [cite_start]turn_phase = f"{primary}" [cite: 32]
                                            
                    [cite_start]if turn_phase == "Neutral": [cite: 33]
                        [cite_start]phase_img_path = "image/turn_phase_neutral.png" [cite: 33]
                    [cite_start]elif turn_phase == "Left": [cite: 33]
                        [cite_start]phase_img_path = "image/turn_phase_left.png" [cite: 33]
                    [cite_start]elif turn_phase == "Right": [cite: 34]
                        [cite_start]phase_img_path = "image/turn_phase_right.png" [cite: 34]
                    [cite_start]else: [cite: 34]
                        [cite_start]phase_img_path = None [cite: 34]

                    [cite_start]connections = [ [cite: 35]
                        ("left_ankle", "left_knee")[cite_start], ("left_knee", "left_hip"), [cite: 35]
                        ("right_ankle", "right_knee")[cite_start], ("right_knee", "right_hip"), [cite: 36]
                        ("left_hip", "right_hip")[cite_start], ("left_shoulder", "right_shoulder"), [cite: 36]
                        ("left_shoulder", "left_elbow")[cite_start], ("right_shoulder", "right_elbow"), [cite: 36]
                        ("right_shoulder", "right_hip")[cite_start], ("left_shoulder", "left_hip"), [cite: 36, 37]
                        ("right_elbow", "right_wrist")[cite_start], ("left_elbow", "left_wrist") [cite: 37]
                    ]
                    [cite_start]for a, b in connections: [cite: 37]
                        [cite_start]if a in joints_2d and b in joints_2d: [cite: 38]
                            [cite_start]pt1, pt2 = joints_2d[a], joints_2d[b] [cite: 38]
                            [cite_start]color = (255, 0, 255) [cite: 38]
                            [cite_start]cv2.line(image, pt1, pt2, color, 2) [cite: 38]

                    [cite_start]for name, (x, y) in joints_2d.items(): [cite: 39]
                        [cite_start]cv2.circle(image, (x, y), 2, (255, 0, 255), -1) [cite: 39]
                    
                    [cite_start]video_resized = resize_keep_aspect(image, target_width=720) [cite: 40]
                    [cite_start]h, w = video_resized.shape[:2] [cite: 40]
                    
                    [cite_start]x_offset = (canvas.shape[1] - w) // 2 [cite: 41]
                    [cite_start]y_offset = 0 [cite: 41]
                    [cite_start]canvas[y_offset:y_offset+h, x_offset:x_offset+w] = video_resized [cite: 41]
                        
                    [cite_start]grid_data = [ [cite: 41]
                        [cite_start]["L-Knee Ext/Flex", safe(left_knee_angle)], [cite: 42]
                        [cite_start]["R-Knee Ext/Flex", safe(right_knee_angle)], [cite: 42]
                        [cite_start]["L-Knee Abd/Add", safe(left_knee_abduction)], [cite: 43]
                        [cite_start]["R-Knee Abd/Add", safe(right_knee_abduction)], [cite: 43]
                        [cite_start]["L-Hip Ext/Flex", safe(left_hip_angle)], [cite: 43]
                        [cite_start]["R-Hip Ext/Flex", safe(right_hip_angle)], [cite: 43, 44]
                        [cite_start]["L-Hip Abd/Add", safe(left_abduction_angle)], [cite: 44]
                        [cite_start]["R-Hip Abd/Add", safe(right_abduction_angle)], [cite: 44]
                        [cite_start]["Torso Tilt", f"{torso_angle:.1f}"], [cite: 44]
                        [cite_start]["Inclination Angle", inclination_display] [cite: 45]
                    ]
                    [cite_start]cell_height = 40 [cite: 45]
                    [cite_start]start_x = 30 [cite: 45]
                    [cite_start]start_y = canvas.shape[0] - len(grid_data)*cell_height - 30 [cite: 45]
                    [cite_start]img_pil = Image.fromarray(canvas) [cite: 46]
                    [cite_start]draw = ImageDraw.Draw(img_pil) [cite: 46]
                    [cite_start]font_path = "static/BestTen-CRT.otf" [cite: 46]
                    [cite_start]font = ImageFont.truetype(font_path, 20) [cite: 46]
                                        
                    [cite_start]for i, (label, value) in enumerate(grid_data): [cite: 47]
                        [cite_start]y_pos = start_y + i * cell_height [cite: 47]
                        [cite_start]top_left = (start_x, start_y + i * 40) [cite: 48]
                        [cite_start]bottom_right = (start_x + 300, start_y + (i + 1) * 40)      [cite: 48]
                        [cite_start]draw.rectangle([top_left, bottom_right], fill=(0,0,0), outline=(255,255,255)) [cite: 48]
                        [cite_start]draw.text((35, y_pos+10), label, font=font, fill=(255,255,255)) [cite: 49]
                        [cite_start]draw.text((200, y_pos+10), value, font=font, fill=(255,255,255)) [cite: 49]
                        
                    [cite_start]canvas = np.array(img_pil) [cite: 49]
                    
                    [cite_start]if phase_img_path and os.path.exists(phase_img_path): [cite: 50]
                        [cite_start]phase_img = cv2.imread(phase_img_path) [cite: 51]
                        [cite_start]if phase_img is not None: [cite: 51]
                            [cite_start]h, w = phase_img.shape[:2] [cite: 51]
                            [cite_start]new_w, new_h = int(w * scale*1.25), int(h * scale*1.25) [cite: 52]
                            [cite_start]phase_resized = cv2.resize(phase_img, (new_w, new_h)) [cite: 52]
                            
                            [cite_start]h, w = phase_resized.shape[:2] [cite: 53]
                            [cite_start]x_offset = 100 [cite: 53]
                            [cite_start]y_offset = canvas.shape[0] // 2 + 55 [cite: 54]
                            [cite_start]canvas[y_offset:y_offset+h, x_offset:x_offset+w] = phase_resized [cite: 54]
                        
                    [cite_start]turn_phase_path = "image/turn_phase.png" [cite: 55]
                    [cite_start]turn_phase = cv2.imread(turn_phase_path) [cite: 55]
                    [cite_start]h, w = turn_phase.shape[:2] [cite: 55]
                    [cite_start]new_w, new_h = int(w * scale*1.25), int(h * scale*1.25) [cite: 55]
                    [cite_start]turn_phase_resized = cv2.resize(turn_phase, (new_w, new_h)) [cite: 56]
                    [cite_start]h, w = turn_phase_resized.shape[:2] [cite: 56]
                    [cite_start]x_offset = 30 [cite: 56]
                    [cite_start]y_offset = canvas.shape[0] // 2 [cite: 57]
                    [cite_start]canvas[y_offset:y_offset+h, x_offset:x_offset+w] = turn_phase_resized [cite: 57]
            else:
                # 骨格未検出時は前フレームの回転履歴をリセット
                prev_rotations = {}
                
                [cite_start]video_resized = resize_keep_aspect(image, target_width=720) [cite: 57, 58]
                [cite_start]h, w = video_resized.shape[:2] [cite: 58]
                [cite_start]x_offset = (canvas.shape[1] - w) // 2 [cite: 58]
                [cite_start]y_offset = 0 [cite: 58]
                [cite_start]canvas[y_offset:y_offset+h, x_offset:x_offset+w] = video_resized [cite: 58]
            
                [cite_start]grid_data = [ [cite: 58, 59]
                    [cite_start]["L-Knee Ext/Flex", "--"], [cite: 59]
                    [cite_start]["R-Knee Ext/Flex", "--"], [cite: 59]
                    [cite_start]["L-Knee Abd/Add", "--"], [cite: 59]
                    [cite_start]["R-Knee Abd/Add", "--"], [cite: 59]
                    [cite_start]["L-Hip Ext/Flex", "--"], [cite: 60]
                    [cite_start]["R-Hip Ext/Flex", "--"], [cite: 60]
                    [cite_start]["L-Hip Abd/Add", "--"], [cite: 60]
                    [cite_start]["R-Hip Abd/Add", "--"], [cite: 60]
                    [cite_start]["Torso Tilt", "--"], [cite: 61]
                    [cite_start]["Inclination Angle", "--"] [cite: 61]
                ] 
                [cite_start]cell_height = 40 [cite: 61]
                [cite_start]start_x = 30 [cite: 61]
                [cite_start]start_y = canvas.shape[0] - len(grid_data)*cell_height - 30 [cite: 61, 62]
                [cite_start]img_pil = Image.fromarray(canvas) [cite: 62]
                [cite_start]draw = ImageDraw.Draw(img_pil) [cite: 62]
                [cite_start]font_path = "static/BestTen-CRT.otf" [cite: 62]
                [cite_start]font = ImageFont.truetype(font_path, 20) [cite: 62]
                           
                [cite_start]for i, (label, value) in enumerate(grid_data): [cite: 63]
                    [cite_start]y_pos = start_y + i * cell_height [cite: 63]
                    [cite_start]top_left = (start_x, start_y + i * 40) [cite: 63]
                    [cite_start]bottom_right = (start_x + 300, start_y + (i + 1) * 40)   [cite: 63, 64]
                    [cite_start]draw.rectangle([top_left, bottom_right], fill=(0,0,0), outline=(255,255,255)) [cite: 64]
                    [cite_start]draw.text((35, y_pos+10), label, font=font, fill=(255,255,255)) [cite: 64]
                    [cite_start]draw.text((200, y_pos+10), value, font=font, fill=(255,255,255)) [cite: 64]
                [cite_start]canvas = np.array(img_pil) [cite: 64]
   
                [cite_start]phase_img_path = "image/turn_phase_none.png" [cite: 65]
                [cite_start]if phase_img_path and os.path.exists(phase_img_path): [cite: 65]
                    [cite_start]phase_img = cv2.imread(phase_img_path) [cite: 65]
                    [cite_start]if phase_img is not None: [cite: 65]
                        [cite_start]h, w = phase_img.shape[:2] [cite: 66]
                        [cite_start]new_w, new_h = int(w * scale*1.25), int(h * scale*1.25) [cite: 66]
                        [cite_start]phase_resized = cv2.resize(phase_img, (new_w, new_h)) [cite: 67]
                            
                        [cite_start]h, w = phase_resized.shape[:2] [cite: 68]
                        [cite_start]x_offset = 120 [cite: 68]
                        [cite_start]y_offset = canvas.shape[0] // 2 + 55 [cite: 68]
                        [cite_start]canvas[y_offset:y_offset+h, x_offset:x_offset+w] = phase_resized [cite: 69]
    
                [cite_start]turn_phase_path = "image/turn_phase.png" [cite: 69]
                [cite_start]turn_phase = cv2.imread(turn_phase_path) [cite: 69]
                [cite_start]h, w = turn_phase.shape[:2] [cite: 69]
                [cite_start]new_w, new_h = int(w * scale*1.25), int(h * scale*1.25) [cite: 69]
                [cite_start]turn_phase_resized = cv2.resize(turn_phase, (new_w, new_h)) [cite: 70]
                [cite_start]h, w = turn_phase_resized.shape[:2] [cite: 70]
                [cite_start]x_offset = 30 [cite: 70]
                [cite_start]y_offset = canvas.shape[0] // 2 [cite: 70]
                [cite_start]canvas[y_offset:y_offset+h, x_offset:x_offset+w] = turn_phase_resized [cite: 70]
                                                           
            [cite_start]out.write(canvas) [cite: 72]
            [cite_start]ret, frame = cap.read() [cite: 72]

    [cite_start]cap.release() [cite: 72]
    [cite_start]out.release() [cite: 72]

    [cite_start]final_output = merge_audio(input_path, temp_output_path) [cite: 72]
    [cite_start]return final_output [cite: 72]
