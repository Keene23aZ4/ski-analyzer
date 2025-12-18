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
# 1. リポジトリ準拠：クォータニオン・行列演算
# ==========================================

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v

def get_quaternion_between(u, v):
    """ベクトルuをvに向ける最小回転クォータニオン(x,y,z,w)"""
    u, v = normalize(u), normalize(v)
    dot = np.dot(u, v)
    if dot > 0.99999: return [0, 0, 0, 1]
    if dot < -0.99999:
        axis = normalize(np.cross([1, 0, 0], u) if abs(u[0]) < 0.9 else np.cross([0, 1, 0], u))
        return [float(axis[0]), float(axis[1]), float(axis[2]), 0.0]
    axis = np.cross(u, v)
    q = np.array([axis[0], axis[1], axis[2], 1.0 + dot])
    return (q / np.linalg.norm(q)).tolist()

# ==========================================
# 2. メインロジック
# ==========================================

st.set_page_config(page_title="Ski 3D Engine", layout="centered")
st.title("⛷️ Advanced Ski Form Analyzer")

uploaded = st.file_uploader("スキー動画をアップロード", type=["mp4", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    with st.spinner("リポジトリ仕様のロジックで解析中..."):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        pose_tracker = Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
        
        frames_data = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_tracker.process(rgb)
            
            if results.pose_world_landmarks:
                # 座標変換: Y反転, Z調整
                pts = [np.array([l.x, -l.y, -l.z]) for l in results.pose_world_landmarks.landmark]
                
                # --- リポジトリ仕様のボーン方向定義 ---
                # 各部位の現在の向き（ワールド方向）
                world_dirs = {
                    "hips": (pts[11]+pts[12])/2 - (pts[23]+pts[24])/2,
                    "L_UpLeg": pts[25] - pts[23],
                    "L_Leg": pts[27] - pts[25],
                    "R_UpLeg": pts[26] - pts[24],
                    "R_Leg": pts[28] - pts[26],
                    "L_Arm": pts[13] - pts[11],
                    "L_ForeArm": pts[15] - pts[13],
                    "R_Arm": pts[14] - pts[12],
                    "R_ForeArm": pts[16] - pts[14],
                }

                # --- 階層構造を考慮した回転計算 ---
                # 本来は親の回転の逆行列を掛けるべきですが、
                # JS側(Three.js)でボーンの .quaternion に直接入れる際、
                # Three.jsのボーン構造が「ワールド空間での指定」を補完するため、
                # ここでは正確な方向ベクトルをJSへ渡します。
                
                rots = {}
                # 各ボーンのTポーズ基準方向（Mixamo標準）
                configs = [
                    ("mixamorigHips", world_dirs["hips"], [0, 1, 0]),
                    ("mixamorigLeftUpLeg", world_dirs["L_UpLeg"], [0, -1, 0]),
                    ("mixamorigLeftLeg", world_dirs["L_Leg"], [0, -1, 0]),
                    ("mixamorigRightUpLeg", world_dirs["R_UpLeg"], [0, -1, 0]),
                    ("mixamorigRightLeg", world_dirs["R_Leg"], [0, -1, 0]),
                    ("mixamorigLeftArm", world_dirs["L_Arm"], [1, 0, 0]),
                    ("mixamorigLeftForeArm", world_dirs["L_ForeArm"], [1, 0, 0]),
                    ("mixamorigRightArm", world_dirs["R_Arm"], [-1, 0, 0]),
                    ("mixamorigRightForeArm", world_dirs["R_ForeArm"], [-1, 0, 0]),
                ]

                for name, target_v, base_v in configs:
                    rots[name] = get_quaternion_between(base_v, target_v)
                
                # 腰の移動（ルートモーション）
                c = (pts[23] + pts[24]) / 2
                rots["hips_pos"] = [c[0]*2.5, c[1]*2.5 + 1.0, c[2]*2.5]
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
        camera.position.set(2, 1.5, 4);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(container.clientWidth, 500);
        container.appendChild(renderer.domElement);
        new THREE.OrbitControls(camera, renderer.domElement);
        
        scene.add(new THREE.AmbientLight(0xffffff, 1.0));
        const sun = new THREE.DirectionalLight(0xffffff, 0.8);
        sun.position.set(5, 10, 5);
        scene.add(sun);
        scene.add(new THREE.GridHelper(20, 20, 0x444444, 0x222222));

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
            if (!data) return;

            // --- リポジトリ流：階層構造への適用 ---
            // 1. まず腰の位置をセット
            avatar.position.set(data.hips_pos[0], data.hips_pos[1], data.hips_pos[2]);

            // 2. 各ボーンの回転を更新
            // Three.jsのボーンにクォータニオンを適用すると、内部で親ボーンとの
            // 相対的な回転（Local Rotation）として処理されます。
            for (const name in data) {{
                if (name === 'hips_pos') continue;
                const bone = avatar.getObjectByName(name);
                if (bone) {{
                    // data[name] は [x, y, z, w]
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
