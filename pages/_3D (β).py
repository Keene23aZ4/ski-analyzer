import streamlit as st
import cv2
import numpy as np
import json
import tempfile
from typing import List, Dict, Any

# MediaPipe imports
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


mp_pose = mp.solutions.pose

st.set_page_config(page_title="3D Pose → Avatar Motion", page_icon="🕺", layout="wide")

# -------------------------
# UI
# -------------------------
st.title("単一動画から3D骨格推定 → 3Dアバターに適用（Streamlit Cloud）")
st.write("1. 動画をアップロード → 2. 3D骨格を推定 → 3. ブラウザで3Dスティックフィギュアを再生")

uploaded = st.file_uploader("動画をアップロード（MP4推奨）", type=["mp4", "mov", "avi", "mkv"])
col_run = st.columns(2)

# Controls
with col_run[0]:
    downsample = st.slider("フレーム間引き（大きいほど軽く）", min_value=1, max_value=10, value=3, step=1)
with col_run[1]:
    show_debug = st.checkbox("処理中の2Dプレビューを表示（遅くなる）", value=False)

# -------------------------
# Helper: process video with MediaPipe Pose
# -------------------------
def extract_3d_pose_sequence(video_path: str, stride: int = 3) -> Dict[str, Any]:
    """
    Returns:
        {
          "landmark_names": [ ... 33 names ... ],
          "frames": [
             { "landmarks": [ {"x":..., "y":..., "z":...}, ... len=33 ] },
             ...
          ],
          "fps": float
        }
    Notes:
      - Uses Pose with model_complexity=2 and enables world landmarks (meters, pelvis-centered)
      - If world landmarks unavailable for a frame, falls back to normalized landmarks and sets z≈0
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("動画を開けませんでした。フォーマットを確認してください。")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False
    )

    frames = []
    frame_idx = 0
    debug_images = []

    # Landmark names for reference
    landmark_names = [lm.name for lm in mp_pose.PoseLandmark]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        # BGR->RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_world_landmarks:
            # world_landmarks are in meters in a real-world scale (origin roughly at pelvis)
            landmarks = result.pose_world_landmarks.landmark
            lm_xyz = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks]
        elif result.pose_landmarks:
            # fallback: normalized image landmarks (x,y in [0,1]); set z=0
            landmarks = result.pose_landmarks.landmark
            lm_xyz = [{"x": lm.x, "y": lm.y, "z": 0.0} for lm in landmarks]
        else:
            # no detection
            lm_xyz = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(len(landmark_names))]

        frames.append({"landmarks": lm_xyz})

        if show_debug and result.pose_landmarks:
            dbg = frame.copy()
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(dbg, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            debug_images.append(cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB))

        frame_idx += 1

    cap.release()
    pose.close()

    return {
        "landmark_names": landmark_names,
        "frames": frames,
        "fps": fps,
        "size": {"width": width, "height": height},
        "debug_images": debug_images
    }

# -------------------------
# Run processing
# -------------------------
if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpf:
        tmpf.write(uploaded.read())
        tmp_path = tmpf.name

    with st.spinner("3D骨格を推定中..."):
        seq = extract_3d_pose_sequence(tmp_path, stride=downsample)

    st.success(f"フレーム数: {len(seq['frames'])} / FPS(推定): {seq['fps']:.2f}")

    if show_debug and seq["debug_images"]:
        st.image(seq["debug_images"], caption="2Dランドマークのプレビュー（間引き後）", use_column_width=True)

    # -------------------------
    # Three.js viewer via HTML component
    # -------------------------
    st.subheader("3Dアバター（スティックフィギュア）再生")
    st.caption("MediaPipeのworld座標（メートル）をThree.js座標に適当にスケール。上半身中心を原点近辺に配置。")

    # Prepare JSON data for JS
    # Normalize/scale a bit for visualization
    # Flip Y/Z to match typical WebGL coordinates: y up, z depth (MediaPipe world y is up; may invert as needed)
    data = {
        "frames": seq["frames"],
        "names": seq["landmark_names"],
        "fps": max(10.0, min(seq["fps"], 60.0)),  # clamp
        "scale": 1.0  # scale in meters → we will re-scale in JS
    }
    payload = json.dumps(data)

    # HTML/JS content
    html = f"""
    <div id="container" style="width:100%; height:600px;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r152/three.min.js"></script>
    <script>
    const payload = {payload};

    // Basic skeleton connections (subset of MediaPipe POSE_CONNECTIONS)
    // Using index order from MediaPipe PoseLandmark enumeration (33 landmarks)
    // This is a simplified set to make a readable stick figure.
    const LINKS = [
      [11, 12], // shoulders
      [11, 13], [13, 15], // left arm
      [12, 14], [14, 16], // right arm
      [23, 24], // hips
      [11, 23], [12, 24], // torso
      [23, 25], [25, 27], // left leg
      [24, 26], [26, 28], // right leg
      [0, 7], [7, 8], [8, 9], [9, 10], // head line approx: nose->eyes->ears (rough)
    ];

    // Scene setup
    const container = document.getElementById('container');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const camera = new THREE.PerspectiveCamera(45, container.clientWidth/container.clientHeight, 0.01, 1000);
    camera.position.set(0, 1.2, 3.0);

    const renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Lights
    const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 0.9);
    hemi.position.set(0, 1, 0);
    scene.add(hemi);

    const dir = new THREE.DirectionalLight(0xffffff, 0.6);
    dir.position.set(5, 5, 5);
    scene.add(dir);

    // Ground grid
    const grid = new THREE.GridHelper(10, 20, 0x444444, 0x222222);
    grid.rotation.x = Math.PI/2; // align for y-up
    scene.add(grid);

    // Stick figure materials
    const jointMat = new THREE.MeshStandardMaterial({color: 0x00e0ff});
    const boneMat = new THREE.LineBasicMaterial({color: 0xffffff});

    // Create joint spheres
    const JOINT_COUNT = 33;
    const joints = [];
    const jointGeom = new THREE.SphereGeometry(0.03, 12, 12);
    for (let i=0; i<JOINT_COUNT; i++) {
      const m = new THREE.Mesh(jointGeom, jointMat);
      scene.add(m);
      joints.push(m);
    }

    // Create bone lines
    const bones = [];
    for (const [a,b] of LINKS) {
      const geom = new THREE.BufferGeometry();
      const positions = new Float32Array(6); // two 3D points
      geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      const line = new THREE.Line(geom, boneMat);
      scene.add(line);
      bones.push({ line, a, b, positions });
    }

    // Orbit controls (optional): lightweight custom orbit
    let isDragging = false, prevX=0, prevY=0, rotX=0, rotY=0;
    renderer.domElement.addEventListener('mousedown', (e)=>{{isDragging=true; prevX=e.clientX; prevY=e.clientY;}});
    renderer.domElement.addEventListener('mouseup', ()=>{{isDragging=false;}});
    renderer.domElement.addEventListener('mousemove', (e)=>{
      if (!isDragging) return;
      const dx = e.clientX - prevX;
      const dy = e.clientY - prevY;
      rotY += dx * 0.005;
      rotX += dy * 0.005;
      prevX = e.clientX; prevY = e.clientY;
    });

    // Animation data
    const frames = payload.frames;
    const fps = payload.fps;
    const dt = 1.0 / fps;

    // Recenter and scale frame landmarks
    function formatFrameLandmarks(frame) {{
      // Choose pelvis center as origin approx: average of left/right hip (23,24)
      const lm = frame.landmarks;
      const hipL = lm[23], hipR = lm[24];
      const cx = (hipL.x + hipR.x) * 0.5;
      const cy = (hipL.y + hipR.y) * 0.5;
      const cz = (hipL.z + hipR.z) * 0.5;

      // Scale to fit better in scene (MediaPipe world units are meters; we compress slightly)
      const S = 1.2;

      // Three.js uses y-up; MediaPipe world landmarks also have y-up.
      // We'll flip X to match intuitive left-right (optional), and keep Z as depth.
      return lm.map(p => {{
        return {{
          x: (p.x - cx) * S,
          y: (p.y - cy) * S,
          z: (p.z - cz) * S
        }};
      }});
    }}

    const cooked = frames.map(formatFrameLandmarks);

    // Animate
    let t=0, idx=0;
    function tick() {{
      requestAnimationFrame(tick);

      // simple orbit
      scene.rotation.y = rotY;
      scene.rotation.x = rotX;

      // advance frame
      t += dt;
      idx = Math.floor(t * fps) % cooked.length;

      const pts = cooked[idx];

      // update joints
      for (let i=0; i<Math.min(JOINT_COUNT, pts.length); i++) {{
        joints[i].position.set(pts[i].x, pts[i].y, pts[i].z);
      }}

      // update bones
      for (const b of bones) {{
        const A = pts[b.a], B = pts[b.b];
        b.positions[0] = A.x; b.positions[1] = A.y; b.positions[2] = A.z;
        b.positions[3] = B.x; b.positions[4] = B.y; b.positions[5] = B.z;
        b.line.geometry.attributes.position.needsUpdate = true;
        b.line.geometry.computeBoundingSphere();
      }}

      renderer.render(scene, camera);
    }}
    tick();

    // Resize handling
    window.addEventListener('resize', ()=>{
      const w = container.clientWidth, h = container.clientHeight;
      camera.aspect = w/h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
    </script>
    """

    st.components.v1.html(html, height=620, scrolling=False)

    st.info("再生中。ドラッグで視点回転できます。間引き量（フレーム間引き）を大きくすると軽くなります。")
else:
    st.warning("MP4などの動画をアップロードしてください。")
