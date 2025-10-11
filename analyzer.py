import cv2
import mediapipe as mp
import numpy as np
import os

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

def process_video(input_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    KEYPOINTS = {
        "nose": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
        "left_foot_index": 31,
        "right_foot_index": 32
    }

    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = os.path.splitext(input_path)[0] + "_processed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while ret:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image = frame.copy()

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            joints = {}
            for name, idx in KEYPOINTS.items():
                x = int(lm[idx].x * width)
                y = int(lm[idx].y * height)
                joints[name] = (x, y)
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

            # 角度計算
            left_knee_angle = calculate_angle(joints["left_hip"], joints["left_knee"], joints["left_ankle"])
            right_knee_angle = calculate_angle(joints["right_hip"], joints["right_knee"], joints["right_ankle"])
            left_hip_angle = calculate_angle(joints["left_knee"], joints["left_hip"], joints["right_hip"])
            right_hip_angle = calculate_angle(joints["right_knee"], joints["right_hip"], joints["left_hip"])
            left_ankle_angle = calculate_angle(joints["left_knee"], joints["left_ankle"], joints["left_foot_index"])
            right_ankle_angle = calculate_angle(joints["right_knee"], joints["right_ankle"], joints["right_foot_index"])

            # 黒文字で角度表示
            angle_texts = [
                f"L-Knee:   {int(left_knee_angle)}°",
                f"R-Knee:   {int(right_knee_angle)}°",
                f"L-Hip:    {int(left_hip_angle)}°",
                f"R-Hip:    {int(right_hip_angle)}°",
                f"L-Ankle:  {int(left_ankle_angle)}°",
                f"R-Ankle:  {int(right_ankle_angle)}°"
            ]
            for i, text in enumerate(angle_texts):
                y_pos = 30 + i * 30
                cv2.putText(image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

            # 線の接続
            connections = [
                ("left_ankle", "left_knee"),
                ("left_knee", "left_hip"),
                ("left_shoulder", "left_elbow"),
                ("left_elbow", "left_wrist"),
                ("right_ankle", "right_knee"),
                ("right_knee", "right_hip"),
                ("right_shoulder", "right_elbow"),
                ("right_elbow", "right_wrist"),
                ("left_hip", "right_hip"),
                ("left_shoulder", "right_shoulder")
            ]
            for a, b in connections:
                pt1 = joints[a]
                pt2 = joints[b]
                cv2.line(image, pt1, pt2, (0, 255, 255), 2)

            # 中点接続線
            shoulder_mid = ((joints["left_shoulder"][0] + joints["right_shoulder"][0]) // 2,
                            (joints["left_shoulder"][1] + joints["right_shoulder"][1]) // 2)
            hip_mid = ((joints["left_hip"][0] + joints["right_hip"][0]) // 2,
                       (joints["left_hip"][1] + joints["right_hip"][1]) // 2)
            cv2.line(image, shoulder_mid, hip_mid, (255, 0, 255), 2)
            cv2.circle(image, shoulder_mid, 5, (255, 0, 255), -1)
            cv2.circle(image, hip_mid, 5, (255, 0, 255), -1)

        out.write(image)
        ret, frame = cap.read()

    cap.release()
    out.release()
    return output_path