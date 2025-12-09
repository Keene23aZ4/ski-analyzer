import streamlit as st
import cv2
import numpy as np
import json
import tempfile
import mediapipe as mp

import base64
from pathlib import Path

font_path = Path(__file__).parent.parent / "static" / "BestTen-CRT.otf"
if font_path.exists():
    encoded = base64.b64encode(font_path.read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        @font-face {{
            font-family: 'BestTen';
            src: url(data:font/opentype;base64,{encoded}) format('opentype');
            font-display: swap;
        }}
        h1, p, div {{
            font-family: 'BestTen', monospace !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 背景画像設定
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


st.set_page_config(page_title="3D Pose → Avatar Motion", page_icon="🕺", layout="wide")

mp_pose = mp.solutions.pose

# -------------------------
# UI
# -------------------------
st.title("単一動画から3D骨格推定 → 3Dアバターに適用（Streamlit Cloud）")
st.write("1. 動画をアップロード → 2. 3D骨格を推定 → 3. ブラウザで3Dスティックフィギュアを再生")

uploaded = st.file_uploader("動画をアップロード（MP4推奨）", type=["mp4", "mov", "avi", "mkv"])
col_run = st.columns(2)

with col_run[0]:
    downsample = st.slider("フレーム間引き（大きいほど軽く）", min_value=1, max_value=10, value=3, step=1)
with col_run[1]:
    show_debug = st.checkbox("処理中の2Dプレビューを表示（遅くなる）", value=False)

# -------------------------
# Helper: process video with MediaPipe Pose
# -------------------------
def extract_3d_pose_sequence(video_path: str, stride: int = 3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("動画を開けませんでした。")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True)

    frames = []
    frame_idx = 0
    debug_images = []
    landmark_names = [lm.name for lm in mp_pose.PoseLandmark]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_world_landmarks:
            landmarks = result.pose_world_landmarks.landmark
            lm_xyz = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks]
        elif result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            lm_xyz = [{"x": lm.x, "y": lm.y, "z": 0.0} for lm in landmarks]
        else:
            lm_xyz = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(len(landmark_names))]

        frames.append({"landmarks": lm_xyz})

        if show_debug and result.pose_landmarks:
            dbg = frame.copy()
            mp.solutions.drawing_utils.draw_landmarks(dbg, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            debug_images.append(cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB))

        frame_idx += 1

    cap.release()
    pose.close()

    return {"landmark_names": landmark_names, "frames": frames, "fps": fps, "debug_images": debug_images}

# -------------------------
# Run processing
# -------------------------
if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpf:
        tmpf.write(uploaded.read())
        tmp_path = tmpf.name

    with st.spinner("3D骨格を推定中..."):
        seq = extract_3d_pose_sequence(tmp_path, stride=downsample)

    st.success(f"フレーム数: {len(seq['frames'])} / FPS: {seq['fps']:.2f}")

    if show_debug and seq["debug_images"]:
        st.image(seq["debug_images"], caption="2Dランドマークプレビュー", use_column_width=True)

    # -------------------------
    # Three.js viewer
    # -------------------------
    st.subheader("3Dアバター（スティックフィギュア）再生")

    data = {
        "frames": seq["frames"],
        "names": seq["landmark_names"],
        "fps": max(10.0, min(seq["fps"], 60.0)),
    }
    payload = json.dumps(data)

    html = f"""
    <div id="container" style="width:100%; height:600px;"></div>
<script src="https://unpkg.com/three@0.152.2/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.152.2/examples/js/controls/OrbitControls.js"></script>
<script>
  // 安定のサイズ取得（clientWidth/Height が 0 の場合に備えて fallback）
  const container = document.getElementById('container');
  const w = container.clientWidth || window.innerWidth;
  const h = container.clientHeight || 600;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111111);

  const camera = new THREE.PerspectiveCamera(60, w / h, 0.01, 1000);
  camera.position.set(0, -1.5, 2.5); // 見上げ視点の初期位置
  camera.lookAt(0, 0.5, 0);

  // f-string 対策でオブジェクトリテラルは {{ ... }} に
  const renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setSize(w, h);
  container.appendChild(renderer.domElement);

  // OrbitControls
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.5, 0);     // 注視点（胴体付近）
  controls.enableDamping = true;      // 慣性
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.9;
  controls.zoomSpeed = 0.9;
  controls.panSpeed = 0.8;

  // 必要に応じた制限（上下の回転など）
  controls.minDistance = 1.0;
  controls.maxDistance = 10.0;
  controls.minPolarAngle = 0.05;      // 真上/真下すぎないように
  controls.maxPolarAngle = Math.PI - 0.05;

  // ライティング
  const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 0.9);
  hemi.position.set(0, 1, 0);
  scene.add(hemi);
  const dir = new THREE.DirectionalLight(0xffffff, 0.6);
  dir.position.set(5, 5, 5);
  scene.add(dir);

  // ここにあなたの joints/bones 生成・更新処理（既存コード）を配置
  // jointMat / boneMat のオブジェクトは {{ ... }} を使う:
  const jointMat = new THREE.MeshStandardMaterial({{color:0x00e0ff}});
  const boneMat  = new THREE.LineBasicMaterial({{color:0xffffff}});
  // ...（略）joints, bones, cooked, tick() など

  // レンダリングループ
  function tick() {
    requestAnimationFrame(tick);
    // 既存のフレーム更新処理（joint/bone の座標更新）をここで実行
    controls.update(); // 慣性を有効化する場合は毎フレーム呼ぶ
    renderer.render(scene, camera);
  }
  tick();

  // リサイズ対応
  window.addEventListener('resize', () => {
    const nw = container.clientWidth || window.innerWidth;
    const nh = container.clientHeight || 600;
    camera.aspect = nw / nh;
    camera.updateProjectionMatrix();
    renderer.setSize(nw, nh);
  });
</script>
    """
    st.components.v1.html(html, height=900, scrolling=False)
else:
    st.warning("MP4などの動画をアップロードしてください。")
