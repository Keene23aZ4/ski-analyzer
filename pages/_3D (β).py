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
# 1. 座標系・回転の修正関数
# ==========================================

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v

def compute_rotation(from_vec, to_vec):
    """
    基準ベクトル(Tポーズ)を目標ベクトル(MediaPipe)へ合わせる回転を計算。
    """
    u, v = normalize(from_vec), normalize(to_vec)
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
# 2. メインアプリ
# ==========================================

st.set_page_config(page_title="Ski 3D Axis Fixed", layout="centered")
st.title("⛷️ Ski Form Synchronized (Axis Fixed)")

uploaded = st.file_uploader("スキー動画をアップロード", type=["mp4", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    with st.spinner("AI骨格解析中..."):
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
                # 【重要】座標変換の修正
                # MediaPipe: x(右正), y(下正), z(手前正)
                # Three.js:  x(右正), y(上正), z(手前正)
                # したがって、yの値だけ符号を反転させると「上下」が正しくなります。
                # また、スキー動画が「自撮り（内カメ）」か「他撮り」かでZ軸の解釈を変える必要があります。
                pts = [np.array([l.x, -l.y, -l.z]) for l in results.pose_world_landmarks.landmark]
                
                # ボーン方向ベクトル
                vectors = {
                    "mixamorigHips": (pts[11]+pts[12])/2 - (pts[23]+pts[24])/2, # 背骨
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
                    # MixamoのTポーズ初期方向の定義
                    if "Arm" in bone: base = [1, 0, 0] if "Left" in bone else [-1, 0, 0]
                    elif "Leg" in bone: base = [0, -1, 0] # 足は下向き
                    else: base = [0, 1, 0] # 体幹は上向き
                    rots[bone] = compute_rotation(base, v)
                
                # Hips（腰）の位置: 倍率をかけて空間内を動かす
                c = (pts[23] + pts[24]) / 2
                rots["hips_pos"] = [c[0]*3, c[1]*3, c[2]*3]
                frames_data.append(rots)
            else:
                frames_data.append(None)
        
        cap.release()
        pose_tracker.close()

    # --- UI/JavaScript セクション ---
    st.subheader("Video & 3D Synchronization")
    
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
        camera.position.set(0, 1.5, 5);
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
                        // Y軸（高さ）に下駄を履かせて地面の上に。
                        avatar.position.set(data[name][0], data[name][1] + 1.2, data[name][2]);
                    }} else {{
                        const bone = avatar.getObjectByName(name);
                        if (bone) bone.quaternion.fromArray(data[name]);
                    }}
                }
            }
        }

        function animate() {{
            requestAnimationFrame(animate);
            updateAvatar();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
    """
    st.components.v1.html(html_code, height=1050)
