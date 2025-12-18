import streamlit as st
import cv2
import numpy as np
import json
import tempfile
import base64
import sys
from pathlib import Path

def set_background():
    img_path = Path(__file__).parent / "static" / "1704273575813.jpg"
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
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #ffffff;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
set_background()

# --- MediaPipe インポートの最終解決策 (Python 3.12 対応) ---
# solutions 属性エラーを回避するために、直接サブモジュールを探索します
try:
    import mediapipe as mp
    # パスを手動で通す
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    Pose = mp_pose.Pose
except (ImportError, AttributeError):
    try:
        # フォールバック: 標準的なインポート
        import mediapipe.solutions.pose as mp_pose
        Pose = mp_pose.Pose
    except Exception as e:
        st.error(f"""
        **MediaPipeの読み込みに失敗しました。**
        エラー内容: {e}
        
        Pythonバージョン: {sys.version}
        
        【対策】
        1. requirements.txt に `mediapipe==0.10.14` と `opencv-python-headless` を記載してください。
        2. Streamlit Cloudの "Reboot app" を実行してキャッシュをクリアしてください。
        """)
        st.stop()

# ==========================================
# 1. 数学・変換ユーティリティ
# ==========================================

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v

def compute_quaternion_from_vectors(u, v):
    u, v = normalize(u), normalize(v)
    dot = np.dot(u, v)
    if dot > 0.999999: return [0, 0, 0, 1]
    if dot < -0.999999:
        axis = normalize(np.cross([1, 0, 0], u) if abs(u[0]) < 0.9 else np.cross([0, 1, 0], u))
        return [float(axis[0]), float(axis[1]), float(axis[2]), 0.0]
    axis = np.cross(u, v)
    q = [float(axis[0]), float(axis[1]), float(axis[2]), float(1.0 + dot)]
    # Normalize Q
    mag = np.sqrt(sum(x*x for x in q))
    return [x/mag for x in q]

# ==========================================
# 2. アプリケーション本体
# ==========================================

st.set_page_config(page_title="Ski 3D Analyzer", layout="wide")
st.title("⛷️ 3D Ski Form Analyzer (Stable Build)")

uploaded = st.file_uploader("スキー動画をアップロードしてください", type=["mp4", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    with st.spinner("AI姿勢解析を実行中..."):
        cap = cv2.VideoCapture(video_path)
        # Python 3.12環境向けにPoseインスタンスを作成
        pose_tracker = Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
        
        frames_data = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_tracker.process(rgb)
            
            if results.pose_world_landmarks:
                # 3D空間のランドマーク (MediaPipe World Landmarks)
                pts = [np.array([l.x, -l.y, -l.z]) for l in results.pose_world_landmarks.landmark]
                
                # スキーのリターゲティングに必要な主要ボーン
                # 各ボーンの方向ベクトルを計算
                bone_vectors = {
                    "mixamorigHips": (pts[11]+pts[12])/2 - (pts[23]+pts[24])/2,
                    "mixamorigLeftUpLeg": pts[25] - pts[23],
                    "mixamorigLeftLeg": pts[27] - pts[25],
                    "mixamorigRightUpLeg": pts[26] - pts[24],
                    "mixamorigRightLeg": pts[28] - pts[26],
                    "mixamorigLeftArm": pts[13] - pts[11],
                    "mixamorigRightArm": pts[14] - pts[12],
                }
                
                frame_rotation = {}
                for bone, vec in bone_vectors.items():
                    # Tポーズの基準方向
                    base_vec = [0, 1, 0] if "Arm" not in bone else ([1, 0, 0] if "Left" in bone else [-1, 0, 0])
                    frame_rotation[bone] = compute_quaternion_from_vectors(base_vec, vec)
                
                frames_data.append(frame_rotation)
        
        cap.release()
        pose_tracker.close()

    # --- Three.js 描画セクション ---
    st.subheader("3Dアバター再生")
    
    model_path = Path("static/avatar.glb")
    if not model_path.exists():
        st.warning("`static/avatar.glb` が見つかりません。デフォルトのボーン構造でプレビューします。")
        model_b64 = ""
    else:
        model_b64 = base64.b64encode(model_path.read_bytes()).decode()

    # JSON データをJSに渡す
    payload = json.dumps(frames_data)

    html_code = f"""
    <div id="container" style="width:100%; height:600px; background-color: #1a1a1a; border-radius: 10px;"></div>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/loaders/GLTFLoader.js"></script>
    
    <script>
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a1a);
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth/600, 0.1, 100);
        camera.position.set(0, 1.5, 4);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, 600);
        document.getElementById('container').appendChild(renderer.domElement);

        scene.add(new THREE.AmbientLight(0xffffff, 0.8));
        const grid = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
        scene.add(grid);

        let avatar;
        if ("{model_b64}") {{
            const loader = new THREE.GLTFLoader();
            loader.load("data:application/octet-stream;base64,{model_b64}", (gltf) => {{
                avatar = gltf.scene;
                scene.add(avatar);
                // ボーン名の正規化（コロン対策）
                avatar.traverse(n => {{ if(n.isBone) n.name = n.name.replace('mixamorig:', 'mixamorig'); }});
            }});
        }}

        const animationData = {payload};
        let frame = 0;

        function animate() {{
            requestAnimationFrame(animate);
            if (avatar && animationData.length > 0) {{
                const current = animationData[frame];
                for (const boneName in current) {{
                    const bone = avatar.getObjectByName(boneName);
                    if (bone) {{
                        const q = current[boneName];
                        bone.quaternion.set(q[0], q[1], q[2], q[3]);
                    }}
                }}
                frame = (frame + 1) % animationData.length;
            }}
            renderer.render(scene, camera);
        }}
        animate();

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / 600;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, 600);
        }});
    </script>
    """
    st.components.v1.html(html_code, height=620)
