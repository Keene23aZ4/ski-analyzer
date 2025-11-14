import cv2
import mediapipe as mp
import numpy as np
import os
import subprocess

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

def calculate_torso_angle(shoulder_mid, hip_mid):
    vector = np.array([hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1]])
    vertical = np.array([0, 1])
    cosine = np.dot(vector, vertical) / (np.linalg.norm(vector) * np.linalg.norm(vertical))
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

def calculate_inclination_angle(center, foot_mid):
    dx = foot_mid[0] - center[0]
    dy = foot_mid[1] - center[1]
    if dy == 0:
        return np.nan
    angle_rad = np.arctan(abs(dx) / abs(dy))
    return np.degrees(angle_rad)

def merge_audio(original_path, processed_path):
    output_path = os.path.splitext(processed_path)[0] + "_with_audio.mp4"
    command = [
        'ffmpeg',
        '-y',
        '-i', original_path,
        '-i', processed_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:a?',
        '-map', '1:v',
        output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

def process_video(input_path, progress_callback=None, show_background=True, selected_angles=None):
    mp_pose = mp.solutions.pose
    KEYPOINTS = {
        "nose": 0,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28
    }

    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_output_path = os.path.splitext(input_path)[0] + "_processed_temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    with mp_pose.Pose() as pose:
        while ret:
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if progress_callback:
                progress_callback(current_frame / total_frames)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            image = frame.copy() if show_background else np.zeros_like(frame)

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

                    def safe(val): return "--" if np.isnan(val) else f"{int(val)}"
                        
                    overlay = image.copy()

                    connections = [
                        ("left_ankle", "left_knee"),
                        ("left_knee", "left_hip"),
                        ("right_ankle", "right_knee"),
                        ("right_knee", "right_hip"),
                        ("left_hip", "right_hip"),
                        ("left_shoulder", "right_shoulder"),
                        ("left_shoulder", "left_wrist"),
                        ("right_shoulder", "right_wrist"),
                        ("right_shoulder", "right_hip"),
                        ("left_shoulder", "left_hip"),
                        ("right_elbow", "right_wrist"),
                        ("left_elbow", "left_wrist")
                    ]
                    for a, b in connections:
                        if a in joints and b in joints:
                            pt1 = joints[a]
                            pt2 = joints[b]
                            color = (255, 0, 255) if (a, b) in [
                                ("left_hip", "right_hip"),
                                ("left_shoulder", "right_shoulder"),
                                ("left_shoulder", "left_hip"),
                                ("right_shoulder", "right_hip")
                            ] else (0, 255, 255)
                            cv2.line(image, pt1, pt2, color, 2)

                    # 関節点の描画
                    for name, (x, y) in joints.items():
                        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                    def safe(val): return "--" if np.isnan(val) else f"{int(val)}°"
                    inclination_display = "--" if np.isnan(inclination＿angle) else f"{inclination_angle:.1f}°"
                    # Turn Phase 判定
                    if np.isnan(inclination_angle):
                        turn_phase = "--"
                    elif inclination_angle <= 10.0:
                        turn_phase = "ニュートラル"
                    else:
                        left_knee_sum = left_knee_abduction + left_knee_angle
                        right_knee_sum = right_knee_abduction + right_knee_angle
                        
                        if left_knee_sum > right_knee_sum:
                            primary = "Left"
                            left_hip_sum = left_hip_angle + left_abduction_angle
                            right_hip_sum = right_hip_angle + right_abduction_angle
                            phase = "First Half" if left_hip_sum > right_hip_sum else "Second Half"
                        else:
                            primary = "Right"
                            right_hip_sum = right_hip_angle + right_abduction_angle
                            left_hip_sum = left_hip_angle + left_abduction_angle
                            phase = "First Half" if right_hip_sum > left_hip_sum else "Second Half"
                        turn_phase = f"{primary} ({phase})"
                        
                    canvas_width = width * 2
                    canvas_height = height
                    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)              
                    canvas[0:height, 0:width] = image

                    # グリッドデータ構築
                    grid_data = [
                        ["L-Knee Ext/Flex", safe(left_knee_angle)],
                        ["R-Knee Ext/Flex", safe(right_knee_angle)],
                        ["L-Knee Abd/Add", safe(left_knee_abduction)],
                        ["R-Knee Abd/Add", safe(right_knee_abduction)],
                        ["L-Hip Ext/Flex", safe(left_hip_angle)],
                        ["R-Hip Ext/Flex", safe(right_hip_angle)],
                        ["L-Hip Abd/Add", safe(left_abduction_angle)],
                        ["R-Hip Abd/Add", safe(right_abduction_angle)],
                        ["Inclination Angle", inclination_display],
                        ["Turn Phase", turn_phase]
                    ]
                    # グリッド描画（右上）
                    cell_width = 180
                    cell_height = 40
                    start_x = width // 2 + 10
                    start_y = 30

                    for i, (label, value) in enumerate(grid_data):
                        top_left = (start_x, start_y + i * cell_height)
                        bottom_right = (start_x + cell_width * 2, start_y + (i + 1) * cell_height)
                        cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), -1)
                        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), 1)
                        cv2.putText(image, label, (top_left[0] + 5, top_left[1] + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                        cv2.putText(image, value, (top_left[0] + cell_width + 5, top_left[1] + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                    # ターンフェーズ表示（左下）
                    cv2.putText(image, f"TURN PHASE: {turn_phase}",
                                (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            out.write(canvas)
            ret, frame = cap.read()

    cap.release()
    out.release()

    final_output = merge_audio(input_path, temp_output_path)
    os.remove(temp_output_path)

    return final_output






