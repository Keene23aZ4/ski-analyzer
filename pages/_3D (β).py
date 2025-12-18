import streamlit as st
import cv2
import numpy as np
import json
import tempfile
import base64
from pathlib import Path


# 背景設定（省略可）
def set_background():
    img_path = Path("static/1704273575813.jpg")
    if img_path.exists():
        encoded = base64.b64encode(img_path.read_bytes()).decode()
        mime = "image/jpeg"
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:{mime};base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
set_background()

# --- MediaPipe 初期化 ---
try:
    import mediapipe as mp
    from mediapipe.python.solutions import pose as mp_pose
    Pose = mp_pose.Pose
except:
    import mediapipe.solutions.pose as mp_pose
    Pose = mp_pose.Pose

# ==========================================
# 1. 高度な数学関数（クォータニオン・行列）
# ==========================================

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v

def quaternion_multiply(q1, q2):
    """クォータニオンの掛け算 (q1 * q2)"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def quaternion_conjugate(q):
    """共役クォータニオン（回転の逆）"""
    x, y, z, w = q
    return np.array([-x, -y, -z, w])

def get_rotation_from_vectors(u, v):
    """ベクトルuをvに合わせる回転Qを計算"""
    u = normalize(u)
    v = normalize(v)
    dot = np.dot(u, v)
    
    if dot > 0.999999: return np.array([0., 0., 0., 1.])
    if dot < -0.999999:
        axis = normalize(np.cross([1, 0, 0], u) if abs(u[0]) < 0.9 else np.cross([0, 1, 0], u))
        return np.array([axis[0], axis[1], axis[2], 0.])
    
    axis = np.cross(u, v)
    q = np.array([axis[0], axis[1], axis[2], 1.0 + dot])
    return normalize(q)

# ==========================================
# 2. メインロジック
# ==========================================

st.set_page_config(page_title="Pro Ski Analyzer", layout="centered")
st.title("⛷️ Precision Ski Analyzer (Hierarchical)")

uploaded = st.file_uploader("スキー動画をアップロード", type=["mp4", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    with st.spinner("物理演算レベルの骨格解析を実行中...（時間がかかります）"):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        # 精度優先設定 (Heavyモデルはエラーが出るため、Standardで平滑化を最大化)
        pose_tracker = Pose(
            static_image_mode=False, 
            model_complexity=1, 
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        frames_data = []
        
        # 親ボーンの定義（階層構造）
        # Child: (Parent, Mixamo_Rest_Vector)
        hierarchy = {
            "hips": (None, [0, 1, 0]), # Root
            "spine": ("hips", [0, 1, 0]),
            "leftUpLeg": ("hips", [0, -1, 0]),
            "rightUpLeg": ("hips", [0, -1, 0]),
            "leftLeg": ("leftUpLeg", [0, -1, 0]),
            "rightLeg": ("rightUpLeg", [0, -1, 0]),
            "leftArm": ("spine", [1, 0, 0]),
            "rightArm": ("spine", [-1, 0, 0]),
            "leftForeArm": ("leftArm", [1, 0, 0]),
            "rightForeArm": ("rightArm", [-1, 0, 0]),
        }

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_tracker.process(rgb)
            
            if results.pose_world_landmarks:
                # 1. 座標変換 (MediaPipe -> Mixamo World)
                lm = results.pose_world_landmarks.landmark
                pts = {}
                # インデックス参照をわかりやすく
                mapping = {
                    11:"L_Shoulder", 12:"R_Shoulder", 
                    23:"L_Hip", 24:"R_Hip",
                    25:"L_Knee", 26:"R_Knee",
                    27:"L_Ankle", 28:"R_Ankle",
                    13:"L_Elbow", 14:"R_Elbow",
                    15:"L_Wrist", 16:"R_Wrist"
                }
                for idx, name in mapping.items():
                    # Y反転, Z調整
                    pts[name] = np.array([lm[idx].x, -lm[idx].y, -lm[idx].z])

                # 2. 現在のボーンベクトルを計算 (Global Vectors)
                # 腰の中心
                center_hip = (pts["L_Hip"] + pts["R_Hip"]) / 2
                center_shoulder = (pts["L_Shoulder"] + pts["R_Shoulder"]) / 2
                
                current_vecs = {
                    "hips": center_shoulder - center_hip, # 脊椎の向きを腰回転とみなす
                    "spine": center_shoulder - center_hip,
                    "leftUpLeg": pts["L_Knee"] - pts["L_Hip"],
                    "rightUpLeg": pts["R_Knee"] - pts["R_Hip"],
                    "leftLeg": pts["L_Ankle"] - pts["L_Knee"],
                    "rightLeg": pts["R_Ankle"] - pts["R_Knee"],
                    "leftArm": pts["L_Elbow"] - pts["L_Shoulder"],
                    "rightArm": pts["R_Elbow"] - pts["R_Shoulder"],
                    "leftForeArm": pts["L_Wrist"] - pts["L_Elbow"],
                    "rightForeArm": pts["R_Wrist"] - pts["R_Elbow"]
                }

                # 3. 階層的回転計算 (Hierarchical Computation)
                global_quats = {} # 各ボーンの「世界に対する」回転
                local_quats = {}  # 親に対する回転 (これをJSに送る)

                for bone_name, (parent_name, rest_vec) in hierarchy.items():
                    # A. グローバル回転を計算 (Tポーズの向き -> 現在の向き)
                    q_global = get_rotation_from_vectors(rest_vec, current_vecs[bone_name])
                    global_quats[bone_name] = q_global
                    
                    # B. ローカル回転に変換
                    if parent_name is None:
                        # ルート(Hips)はグローバル回転そのもの
                        local_quats["mixamorigHips"] = q_global.tolist()
                    else:
                        # Q_local = inverse(Q_parent_global) * Q_current_global
                        q_parent_inv = quaternion_conjugate(global_quats[parent_name])
                        q_local = quaternion_multiply(q_parent_inv, q_global)
                        # 正規化
                        q_local = normalize(q_local)
                        
                        # 名前変換 (内部用 -> Mixamo用)
                        mixamo_name = "mixamorig" + bone_name[0].upper() + bone_name[1:]
                        local_quats[mixamo_name] = q_local.tolist()

                # 腰の位置補正
                local_quats["hips_pos"] = [center_hip[0]*2.2, center_hip[1]*2.2 + 1.05, center_hip[2]*2.2]
                
                frames_data.append(local_quats)
            else:
                frames_data.append(None)
        cap.release()
        pose_tracker.close()

    # --- UI & Viewer ---
    video_bytes = open(video_path, 'rb').read()
    video_b64 = base64.b64encode(video_bytes).decode()
    model_path = Path("static/avatar.glb")
    model_b64 = base64.b64encode(model_path.read_bytes()).decode() if model_path.exists() else ""
    payload = json.dumps({"fps": fps, "frames": frames_data})

    html_code = f"""
    <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
        <video id="sync_video" width="100%" controls playsinline style="border-radius: 8px;">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        <div id="container" style="width:100%; height:600px; background:#111; border-radius:8px;"></div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        const video = document.getElementById('sync_video');
        const container = document.getElementById('container');
        const animData = {payload};
        
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, container.clientWidth/600, 0.1, 100);
        camera.position.set(0, 1.5, 4.5);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(container.clientWidth, 600);
        container.appendChild(renderer.domElement);
        new THREE.OrbitControls(camera, renderer.domElement);
        
        scene.add(new THREE.AmbientLight(0xffffff, 1.0));
        const grid = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
        scene.add(grid);

        let avatar;
        new THREE.GLTFLoader().load("data:application/octet-stream;base64,{model_b64}", (gltf) => {{
            avatar = gltf.scene;
            scene.add(avatar);
            avatar.traverse(n => {{ 
                if(n.isBone) n.name = n.name.replace('mixamorig:', 'mixamorig'); 
            }});
        }});

        function update() {{
            if (!avatar || !animData.frames.length) return;
            let fIdx = Math.floor(video.currentTime * animData.fps);
            if (fIdx >= animData.frames.length) fIdx = animData.frames.length - 1;
            
            const data = animData.frames[fIdx];
            if (data) {{
                // 1. 位置更新
                avatar.position.set(data.hips_pos[0], data.hips_pos[1], data.hips_pos[2]);
                
                // 2. 回転更新 (Local Rotation)
                for (const name in data) {{
                    if (name === 'hips_pos') continue;
                    const bone = avatar.getObjectByName(name);
                    if (bone) {{
                        // アニメーションの補間（Slerp）を行うとさらに滑らかになりますが
                        // ここでは正確性を重視してフレームの生の計算値を適用
                        bone.quaternion.set(data[name][0], data[name][1], data[name][2], data[name][3]);
                    }}
                }}
            }}
        }}

        function animate() {{
            requestAnimationFrame(animate);
            update();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
    """
    st.components.v1.html(html_code, height=1150)
