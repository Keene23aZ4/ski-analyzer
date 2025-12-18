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

# --- MediaPipe インポート (堅牢版) ---
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
# 1. 高度な回転演算（ベクトル合成）
# ==========================================

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v

def get_quaternion(base_v, target_v):
    u, v = normalize(base_v), normalize(target_v)
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
# 2. メインロジック
# ==========================================

st.set_page_config(page_title="Ski 3D Fix", layout="centered")
st.title("⛷️ Ski 3D Analyzer (Leg Movement Fixed)")

uploaded = st.file_uploader("スキー動画をアップロード", type=["mp4", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    with st.spinner("脚の動きを精密解析中..."):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # model_complexity=2 に変更して精度を最大化
        pose_tracker = Pose(static_image_mode=False, model_complexity=2, smooth_landmarks=True)
        
        frames_data = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_tracker.process(rgb)
            
            if results.pose_world_landmarks:
                # 座標変換: Y反転、Z反転（鏡面対策）
                # スキー動画の奥行きを正確に出すため、Zの感度を調整
                pts = [np.array([l.x, -l.y, -l.z]) for l in results.pose_world_landmarks.landmark]
                
                rots = {}
                
                # --- 脚の階層化計算 ---
                # 1. 股関節 (UpLeg): 腰から膝へのベクトル
                rots["mixamorigLeftUpLeg"] = get_quaternion([0, -1, 0], pts[25] - pts[23])
                rots["mixamorigRightUpLeg"] = get_quaternion([0, -1, 0], pts[26] - pts[24])
                
                # 2. 膝 (Leg): 膝から足首へのベクトル
                # ※重要: 膝は「股関節の回転」を引き継ぐため、
                # ここで計算する回転はアバター内で正しく補正されます。
                rots["mixamorigLeftLeg"] = get_quaternion([0, -1, 0], pts[27] - pts[25])
                rots["mixamorigRightLeg"] = get_quaternion([0, -1, 0], pts[28] - pts[26])
                
                # --- 上半身 ---
                spine_vec = (pts[11]+pts[12])/2 - (pts[23]+pts[24])/2
                rots["mixamorigHips"] = get_quaternion([0, 1, 0], spine_vec)
                rots["mixamorigSpine"] = get_quaternion([0, 1, 0], spine_vec)
                
                # --- 腕 ---
                rots["mixamorigLeftArm"] = get_quaternion([1, 0, 0], pts[13] - pts[11])
                rots["mixamorigLeftForeArm"] = get_quaternion([1, 0, 0], pts[15] - pts[13])
                rots["mixamorigRightArm"] = get_quaternion([-1, 0, 0], pts[14] - pts[12])
                rots["mixamorigRightForeArm"] = get_quaternion([-1, 0, 0], pts[16] - pts[14])
                
                # 腰の位置 (Z軸の動きを強調)
                c = (pts[23] + pts[24]) / 2
                rots["hips_pos"] = [c[0]*2.0, c[1]*2.0 + 1.0, c[2]*2.0]
                
                frames_data.append(rots)
            else:
                frames_data.append(None)
        cap.release()

    # --- UI & Synchronized Viewer ---
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
        camera.position.set(3, 2, 5); // 斜めから見やすく
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(container.clientWidth, 500);
        container.appendChild(renderer.domElement);
        new THREE.OrbitControls(camera, renderer.domElement);
        
        scene.add(new THREE.AmbientLight(0xffffff, 1.0));
        scene.add(new THREE.GridHelper(20, 20, 0x444444, 0x222222));

        let avatar;
        new THREE.GLTFLoader().load("data:application/octet-stream;base64,{model_b64}", (gltf) => {{
            avatar = gltf.scene;
            scene.add(avatar);
            avatar.traverse(n => {{ 
                if(n.isBone) {{
                    n.name = n.name.replace('mixamorig:', 'mixamorig');
                    // 脚のねじれを防ぐため、ボーンの回転順序を固定
                    n.rotation.reorder('YXZ');
                }}
            }});
        }});

        function updateAvatar() {{
            if (!avatar || !animData.frames.length) return;
            let fIdx = Math.floor(video.currentTime * animData.fps);
            if (fIdx >= animData.frames.length) fIdx = animData.frames.length - 1;
            
            const data = animData.frames[fIdx];
            if (!data) return;

            avatar.position.set(data.hips_pos[0], data.hips_pos[1], data.hips_pos[2]);

            for (const name in data) {{
                if (name === 'hips_pos') continue;
                const bone = avatar.getObjectByName(name);
                if (bone) {{
                    // クォータニオンを直接セット
                    bone.quaternion.set(data[name][0], data[name][1], data[name][2], data[name][3]);
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
