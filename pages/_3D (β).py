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
# 1. 座標・回転計算 (リターゲティング・ロジック)
# ==========================================

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v

def get_quaternion(u, v):
    """ベクトルuをベクトルvに回転させる最短回転クォータニオンを計算"""
    u = normalize(u)
    v = normalize(v)
    dot = np.dot(u, v)
    if dot > 0.99999: return [0, 0, 0, 1]
    if dot < -0.99999:
        axis = normalize(np.cross([1, 0, 0], u) if abs(u[0]) < 0.9 else np.cross([0, 1, 0], u))
        return [float(axis[0]), float(axis[1]), float(axis[2]), 0.0]
    axis = np.cross(u, v)
    q = np.array([axis[0], axis[1], axis[2], 1.0 + dot])
    mag = np.linalg.norm(q)
    return (q / mag).tolist()

# ==========================================
# 2. メインアプリケーション
# ==========================================

st.set_page_config(page_title="Ski 3D Mocap", layout="centered")
st.title("⛷️ Professional Ski 3D Analyzer")

uploaded = st.file_uploader("スキー動画をアップロード", type=["mp4", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    with st.spinner("AI骨格抽出中..."):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        pose_tracker = Pose(static_image_mode=False, model_complexity=1)
        
        frames_data = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_tracker.process(rgb)
            
            if results.pose_world_landmarks:
                # MediaPipe: x(右), y(下), z(奥) -> Three.js: x(右), y(上), z(手前)
                # yを反転させ、zの符号を調整して右手系に合わせます
                pts = [np.array([l.x, -l.y, -l.z]) for l in results.pose_world_landmarks.landmark]
                
                # ボーンごとのターゲットベクトルを定義
                vectors = {
                    "mixamorigHips": (pts[11]+pts[12])/2 - (pts[23]+pts[24])/2,
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
                
                rots = {}
                for bone, v in vectors.items():
                    # MixamoのTポーズ初期方向 (基準)
                    if "Arm" in bone: base = [1, 0, 0] if "Left" in bone else [-1, 0, 0]
                    elif "Leg" in bone: base = [0, -1, 0]
                    else: base = [0, 1, 0]
                    rots[bone] = get_quaternion(base, v)
                
                # 腰の位置 (正規化された座標に倍率をかける)
                c = (pts[23] + pts[24]) / 2
                rots["hips_pos"] = [c[0]*2, c[1]*2 + 1.1, c[2]*2]
                frames_data.append(rots)
            else:
                frames_data.append(None)
        cap.release()

    # --- UI & Synchronized Viewer ---
    st.subheader("Video & 3D Sync View")
    
    video_bytes = open(video_path, 'rb').read()
    video_b64 = base64.b64encode(video_bytes).decode()
    
    model_path = Path("static/avatar.glb")
    model_b64 = base64.b64encode(model_path.read_bytes()).decode() if model_path.exists() else ""
    
    payload = json.dumps({"fps": fps, "frames": frames_data})

    # Pythonのf-string内でのJavaScript波括弧エラー回避のため {{ }} を使用
    html_code = f"""
    <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
        <video id="sync_video" width="100%" controls playsinline style="border-radius: 8px;">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        <div id="container" style="width:100%; height:500px; background:#111; border-radius:8px;"></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        const video = document.getElementById('sync_video');
        const container = document.getElementById('container');
        const animData = {payload};
        
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, container.clientWidth/500, 0.1, 100);
        camera.position.set(2, 2, 5);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(container.clientWidth, 500);
        container.appendChild(renderer.domElement);
        new THREE.OrbitControls(camera, renderer.domElement);
        
        scene.add(new THREE.AmbientLight(0xffffff, 1.2));
        scene.add(new THREE.GridHelper(10, 10, 0x444444, 0x222222));

        let avatar;
        new THREE.GLTFLoader().load("data:application/octet-stream;base64,{model_b64}", (gltf) => {{
            avatar = gltf.scene;
            scene.add(avatar);
            // Mixamoのボーン名コロン対策
            avatar.traverse(n => {{ 
                if(n.isBone) n.name = n.name.replace('mixamorig:', 'mixamorig'); 
            }});
        }});

        function updateAvatar() {{
            if (!avatar || !animData.frames.length) return;
            let fIdx = Math.floor(video.currentTime * animData.fps);
            if (fIdx >= animData.frames.length) fIdx = animData.frames.length - 1;
            
            const data = animData.frames[fIdx];
            if (data) {{
                for (const name in data) {{
                    if (name === 'hips_pos') {{
                        avatar.position.set(data[name][0], data[name][1], data[name][2]);
                    }} else {{
                        // ボーン名が一致するか、接頭辞なしでも探す
                        const bone = avatar.getObjectByName(name) || avatar.getObjectByName(name.replace('mixamorig', ''));
                        if (bone) bone.quaternion.fromArray(data[name]);
                    }}
                }}
            }}
        }}

        function animate() {{
            requestAnimationFrame(animate);
            updateAvatar();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
    """
    st.components.v1.html(html_code, height=1050)
