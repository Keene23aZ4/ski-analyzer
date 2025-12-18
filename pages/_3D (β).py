import streamlit as st
import cv2
import numpy as np
import json
import tempfile
import base64
import sys
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

# --- MediaPipe インポート (Python 3.12 堅牢版) ---
try:
    import mediapipe as mp
    from mediapipe.python.solutions import pose as mp_pose
    Pose = mp_pose.Pose
except (ImportError, AttributeError):
    try:
        import mediapipe.solutions.pose as mp_pose
        Pose = mp_pose.Pose
    except Exception as e:
        st.error(f"MediaPipe読み込みエラー: {e}")
        st.stop()

# ==========================================
# 1. 高精度クォータニオン計算
# ==========================================

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v

def compute_rotation(from_vec, to_vec):
    """from_vec を to_vec に向ける最短回転クォータニオンを計算"""
    u = normalize(from_vec)
    v = normalize(to_vec)
    dot = np.dot(u, v)
    
    if dot > 0.99999:
        return [0, 0, 0, 1]
    if dot < -0.99999:
        # 180度回転の場合は任意の直交軸を回転軸にする
        axis = normalize(np.cross([1, 0, 0], u) if abs(u[0]) < 0.9 else np.cross([0, 1, 0], u))
        return [float(axis[0]), float(axis[1]), float(axis[2]), 0.0]
    
    axis = np.cross(u, v)
    w = 1.0 + dot
    q = np.array([axis[0], axis[1], axis[2], w])
    mag = np.linalg.norm(q)
    return (q / mag).tolist()

# ==========================================
# 2. メインロジック
# ==========================================

st.set_page_config(page_title="Ski 3D Analyzer Pro", layout="wide")
st.title("⛷️ 3D Ski Form Analyzer (Motion Fixed)")

uploaded = st.file_uploader("動画をアップロード", type=["mp4", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    with st.spinner("骨格解析中..."):
        cap = cv2.VideoCapture(video_path)
        pose_tracker = Pose(static_image_mode=False, model_complexity=1)
        
        frames_data = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_tracker.process(rgb)
            
            if results.pose_world_landmarks:
                # MediaPipe: x(右), y(下), z(手前)
                # Three.js:  x(右), y(上), z(奥) に合わせるため Y, Z を反転
                pts = [np.array([l.x, -l.y, -l.z]) for l in results.pose_world_landmarks.landmark]
                
                # スキーに必要な主要関節ベクトル
                # MixamoのTポーズは：腕は横(X)、足は下(-Y)、背骨は上(Y)
                vectors = {
                    "mixamorigHips": (pts[11]+pts[12])/2 - (pts[23]+pts[24])/2, # 脊椎方向
                    "mixamorigSpine": (pts[11]+pts[12])/2 - (pts[23]+pts[24])/2,
                    "mixamorigLeftUpLeg": pts[25] - pts[23],
                    "mixamorigLeftLeg": pts[27] - pts[25],
                    "mixamorigRightUpLeg": pts[26] - pts[24],
                    "mixamorigRightLeg": pts[28] - pts[26],
                    "mixamorigLeftArm": pts[13] - pts[11],
                    "mixamorigLeftForeArm": pts[15] - pts[13],
                    "mixamorigRightArm": pts[14] - pts[12],
                    "mixamorigRightForeArm": pts[16] - pts[14],
                }
                
                rotations = {}
                for bone, v in vectors.items():
                    # 各ボーンの初期方向（Tポーズ基準）
                    if "Arm" in bone:
                        base = [1, 0, 0] if "Left" in bone else [-1, 0, 0]
                    elif "Leg" in bone:
                        base = [0, -1, 0]
                    else:
                        base = [0, 1, 0] # Spine/Hips
                    
                    rotations[bone] = compute_rotation(base, v)
                
                # 腰のグローバル位置（少しスケーリング）
                center_hips = (pts[23] + pts[24]) / 2
                rotations["hips_pos"] = [center_hips[0]*2, center_hips[1]*2, center_hips[2]*2]
                
                frames_data.append(rotations)
        
        cap.release()

    # --- Three.js ビジュアライザー ---
    model_path = Path("static/avatar.glb")
    model_b64 = base64.b64encode(model_path.read_bytes()).decode() if model_path.exists() else ""

    html_code = f"""
    <div id="container" style="width:100%; height:600px; background:#111;"></div>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth/600, 0.1, 100);
        camera.position.set(0, 1.5, 5);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, 600);
        document.getElementById('container').appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        scene.add(new THREE.AmbientLight(0xffffff, 1.0));
        scene.add(new THREE.GridHelper(10, 10));

        let model;
        const loader = new THREE.GLTFLoader();
        loader.load("data:application/octet-stream;base64,{model_b64}", (gltf) => {{
            model = gltf.scene;
            scene.add(model);
            model.traverse(n => {{ 
                if(n.isBone) n.name = n.name.replace('mixamorig:', 'mixamorig'); 
            }});
        }});

        const frames = {json.dumps(frames_data)};
        let fIdx = 0;

        function animate() {{
            requestAnimationFrame(animate);
            if (model && frames.length > 0) {{
                const data = frames[fIdx];
                for (const name in data) {{
                    if (name === 'hips_pos') {{
                        model.position.set(data[name][0], data[name][1] + 1.0, data[name][2]);
                    }} else {{
                        const bone = model.getObjectByName(name);
                        if (bone) bone.quaternion.fromArray(data[name]);
                    }}
                }}
                fIdx = (fIdx + 1) % frames.length;
            }}
            renderer.render(scene, camera);
        }}
        animate();
    </script>
    """
    st.components.v1.html(html_code, height=620)
