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

# --- MediaPipe インポートの最終解決策 ---
try:
    import mediapipe as mp
    # solutions が直接参照できない場合があるため、以下の記述で統一します
    BasePose = mp.solutions.pose.Pose
    mp_pose_mod = mp.solutions.pose
except Exception as e:
    st.error(f"MediaPipeの読み込みに失敗しました。requirements.txtを確認してください: {e}")
    st.stop()

# ==========================================
# 1. ユーティリティ（GLB解析・数学）
# ==========================================

def extract_default_dirs(glb_path):
    """
    リターゲティングに必須の関数。
    GLBからボーンの初期方向（Tポーズベクトル）を抽出します。
    """
    # 本来は trimesh 等で解析しますが、簡易化のため標準的な値を返します。
    # ここが実際のボーン構造と一致することが重要です。
    return {
        "mixamorigHips": [0, 1, 0],
        "mixamorigSpine": [0, 1, 0],
        "mixamorigSpine1": [0, 1, 0],
        "mixamorigSpine2": [0, 1, 0],
        "mixamorigNeck": [0, 1, 0],
        "mixamorigHead": [0, 1, 0],
        "mixamorigLeftArm": [1, 0, 0],
        "mixamorigLeftForeArm": [1, 0, 0],
        "mixamorigLeftHand": [1, 0, 0],
        "mixamorigRightArm": [-1, 0, 0],
        "mixamorigRightForeArm": [-1, 0, 0],
        "mixamorigRightHand": [-1, 0, 0],
        "mixamorigLeftUpLeg": [0, -1, 0],
        "mixamorigLeftLeg": [0, -1, 0],
        "mixamorigLeftFoot": [0, -1, 0],
        "mixamorigRightUpLeg": [0, -1, 0],
        "mixamorigRightLeg": [0, -1, 0],
        "mixamorigRightFoot": [0, -1, 0],
    }

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v

def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=float)

def quat_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)

def compute_quaternion_from_vectors(u, v):
    u, v = normalize(u), normalize(v)
    dot = np.dot(u, v)
    if dot > 0.999999: return np.array([0, 0, 0, 1], dtype=float)
    if dot < -0.999999:
        axis = normalize(np.cross([1, 0, 0], u) if abs(u[0]) < 0.9 else np.cross([0, 1, 0], u))
        return np.array([axis[0], axis[1], axis[2], 0], dtype=float)
    axis = np.cross(u, v)
    q = np.array([axis[0], axis[1], axis[2], 1.0 + dot])
    return normalize(q)

# ==========================================
# 2. 定数・階層定義
# ==========================================

HIERARCHY = {
    "mixamorigHips": None,
    "mixamorigSpine": "mixamorigHips",
    "mixamorigSpine1": "mixamorigSpine",
    "mixamorigSpine2": "mixamorigSpine1",
    "mixamorigNeck": "mixamorigSpine2",
    "mixamorigLeftArm": "mixamorigSpine2",
    "mixamorigLeftForeArm": "mixamorigLeftArm",
    "mixamorigRightArm": "mixamorigSpine2",
    "mixamorigRightForeArm": "mixamorigRightArm",
    "mixamorigLeftUpLeg": "mixamorigHips",
    "mixamorigLeftLeg": "mixamorigLeftUpLeg",
    "mixamorigRightUpLeg": "mixamorigHips",
    "mixamorigRightLeg": "mixamorigRightUpLeg",
}

PROCESS_ORDER = ["mixamorigHips", "mixamorigSpine", "mixamorigSpine1", "mixamorigSpine2", 
                 "mixamorigNeck", "mixamorigLeftArm", "mixamorigLeftForeArm", 
                 "mixamorigRightArm", "mixamorigRightForeArm",
                 "mixamorigLeftUpLeg", "mixamorigLeftLeg", "mixamorigRightUpLeg", "mixamorigRightLeg"]

# ==========================================
# 3. Streamlit アプリ
# ==========================================

st.set_page_config(page_title="Ski 3D Analyzer", layout="wide")

def extract_3d_pose_sequence(video_path, stride=3):
    cap = cv2.VideoCapture(video_path)
    pose = BasePose(model_complexity=1, smooth_landmarks=True)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_world_landmarks:
            frames.append({"landmarks": [{"x": l.x, "y": l.y, "z": l.z} for l in res.pose_world_landmarks.landmark]})
    cap.release()
    return frames

uploaded = st.file_uploader("動画をアップロード", type=["mp4", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    with st.spinner("解析中..."):
        frames_raw = extract_3d_pose_sequence(video_path)
        
        # モーション変換ロジック
        default_dirs = extract_default_dirs("") # GLB解析を代替
        anim_frames = []
        for f in frames_raw:
            pts = [np.array([l["x"], -l["y"], -l["z"]]) for l in f["landmarks"]]
            
            # ボーンごとのベクトル算出（主要部位のみ）
            current_vectors = {
                "mixamorigHips": (pts[11]+pts[12])/2 - (pts[23]+pts[24])/2,
                "mixamorigLeftArm": pts[13] - pts[11],
                "mixamorigLeftForeArm": pts[15] - pts[13],
                "mixamorigRightArm": pts[14] - pts[12],
                "mixamorigRightForeArm": pts[16] - pts[14],
                "mixamorigLeftUpLeg": pts[25] - pts[23],
                "mixamorigLeftLeg": pts[27] - pts[25],
                "mixamorigRightUpLeg": pts[26] - pts[24],
                "mixamorigRightLeg": pts[28] - pts[26],
            }
            
            fd = {}
            global_quats = {}
            for bone in PROCESS_ORDER:
                def_v = np.array(default_dirs.get(bone, [0,1,0]))
                tgt_v = current_vectors.get(bone, def_v)
                q_global = compute_quaternion_from_vectors(def_v, tgt_v)
                
                parent = HIERARCHY.get(bone)
                if parent and parent in global_quats:
                    q_local = quat_multiply(quat_conjugate(global_quats[parent]), q_global)
                else:
                    q_local = q_global
                
                global_quats[bone] = q_global
                fd[bone] = q_local.tolist()
            
            fd["mixamorigHips_pos"] = ((pts[23] + pts[24]) / 2).tolist()
            anim_frames.append(fd)

    # HTML出力（Three.js 連携）
    payload = json.dumps({"frames": anim_frames})
    model_path = Path("static/avatar.glb")
    model_b64 = base64.b64encode(model_path.read_bytes()).decode() if model_path.exists() else ""

    st.components.v1.html(f"""
        <div id="container" style="width:100%; height:500px; background:#111;"></div>
        <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/loaders/GLTFLoader.js"></script>
        <script>
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(45, window.innerWidth/500, 0.1, 100);
            camera.position.set(0, 1, 3);
            const renderer = new THREE.WebGLRenderer({{antialias:true}});
            renderer.setSize(window.innerWidth, 500);
            document.getElementById('container').appendChild(renderer.domElement);
            scene.add(new THREE.AmbientLight(0xffffff, 0.8));
            
            let avatar;
            const loader = new THREE.GLTFLoader();
            loader.load("data:application/octet-stream;base64,{model_b64}", (gltf) => {{
                avatar = gltf.scene;
                scene.add(avatar);
            }});

            const anim = {payload};
            let frameIdx = 0;
            function animate() {{
                requestAnimationFrame(animate);
                if(avatar && anim.frames.length > 0) {{
                    const frame = anim.frames[frameIdx];
                    for(const b in frame) {{
                        const bone = avatar.getObjectByName(b) || avatar.getObjectByName(b.replace('mixamorig', 'mixamorig:'));
                        if(bone && !b.endsWith('_pos')) bone.quaternion.fromArray(frame[b]);
                    }}
                    frameIdx = (frameIdx + 1) % anim.frames.length;
                }}
                renderer.render(scene, camera);
            }}
            animate();
        </script>
    """, height=550)

    # プレースホルダー置換
    html_code = html_code.replace("PAYLOAD_JSON", payload)
    html_code = html_code.replace("MODEL_B64", model_data)
    html_code = html_code.replace("VIDEO_B64", video_data)

    st.components.v1.html(html_code, height=750, scrolling=False)
