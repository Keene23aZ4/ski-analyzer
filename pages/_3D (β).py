import streamlit as st
import cv2
import numpy as np
import json
import tempfile
import mediapipe as mp
import base64
from pathlib import Path

from retarget import extract_default_dirs  # ここで GLB から default_dir を取る

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

st.set_page_config(page_title="3D Pose → Avatar Motion", layout="wide")

mp_pose = mp.solutions.pose

def extract_3d_pose_sequence(video_path: str, stride: int = 3):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True)

    frames = []
    landmark_names = [lm.name for lm in mp_pose.PoseLandmark]
    frame_idx = 0

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
        else:
            lm_xyz = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(len(landmark_names))]

        frames.append({"landmarks": lm_xyz})
        frame_idx += 1

    cap.release()
    pose.close()
    return {"landmark_names": landmark_names, "frames": frames, "fps": fps}


uploaded = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi", "mkv"])
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpf:
        tmpf.write(uploaded.read())
        tmp_path = tmpf.name

    seq = extract_3d_pose_sequence(tmp_path, stride=3)

    # GLB から Tポーズの default_dir を取得（コロン付き Mixamo 名と対応）
    model_path = Path("static/avatar.glb")
    default_dirs = extract_default_dirs(str(model_path))

    # MediaPipe のランドマーク index と Mixamo のボーン名対応
    # （※ここは元のマッピングそのまま。必要なら後で微調整）
    MIXAMO_MAP = {
        "mixamorig:Hips": 23,          # LEFT_HIP をルートとして扱う
        "mixamorig:LeftArm": 11,
        "mixamorig:LeftForeArm": 13,
        "mixamorig:LeftHand": 15,
        "mixamorig:RightArm": 12,
        "mixamorig:RightForeArm": 14,
        "mixamorig:RightHand": 16,
        "mixamorig:LeftUpLeg": 23,    # hips と同じ起点
        "mixamorig:LeftLeg": 25,
        "mixamorig:LeftFoot": 27,
        "mixamorig:LeftToeBase": 31,
        "mixamorig:RightUpLeg": 24,
        "mixamorig:RightLeg": 26,
        "mixamorig:RightFoot": 28,
        "mixamorig:RightToeBase": 32,
    }

    def mp_to_mixamo_vec(lm):
        # MediaPipe → Three.js / Mixamo の座標系ざっくり合わせ
        return np.array([lm["x"], -lm["y"], -lm["z"]], dtype=float)

    def compute_spine_points(hips, neck):
        spine1 = (hips + neck) / 2
        spine = (hips + spine1) / 2
        spine2 = (spine1 + neck) / 2
        return spine, spine1, spine2

    def compute_quaternion(default_dir, target_dir):
        default_dir = default_dir / np.linalg.norm(default_dir)
        target_dir = target_dir / np.linalg.norm(target_dir)

        axis = np.cross(default_dir, target_dir)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-6:
            return [0, 0, 0, 1]

        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(default_dir, target_dir), -1.0, 1.0))

        qw = np.cos(angle / 2)
        qx, qy, qz = axis * np.sin(angle / 2)
        return [qx, qy, qz, qw]

    def convert_to_mixamo_json(frames):
        anim = {"frames": []}

        for f in frames:
            LM = f["landmarks"]
            pts = [mp_to_mixamo_vec(lm) for lm in LM]

            # Hips / Neck / Head の基準（ここはシンプルにまず鼻を neck/head として使う）
            hips = pts[23]
            neck = pts[0]
            head = pts[0]

            spine, spine1, spine2 = compute_spine_points(hips, neck)

            frame_data = {}
            frame_data["mixamorig:Hips_pos"] = hips.tolist()

            # --- Spine 系の回転 ---
            # default_dir を GLB から取っているので、direction ベクトルだけ与える
            def_dir_spine  = np.array(default_dirs.get("mixamorig:Spine",  [0, 1, 0]), dtype=float)
            def_dir_spine1 = np.array(default_dirs.get("mixamorig:Spine1", [0, 1, 0]), dtype=float)
            def_dir_spine2 = np.array(default_dirs.get("mixamorig:Spine2", [0, 1, 0]), dtype=float)
            def_dir_neck   = np.array(default_dirs.get("mixamorig:Neck",   [0, 1, 0]), dtype=float)
            def_dir_head   = np.array(default_dirs.get("mixamorig:Head",   [0, 1, 0]), dtype=float)

            frame_data["mixamorig:Spine"]  = compute_quaternion(def_dir_spine,  spine - hips)
            frame_data["mixamorig:Spine1"] = compute_quaternion(def_dir_spine1, spine1 - spine)
            frame_data["mixamorig:Spine2"] = compute_quaternion(def_dir_spine2, spine2 - spine1)
            frame_data["mixamorig:Neck"]   = compute_quaternion(def_dir_neck,   neck - spine2)
            frame_data["mixamorig:Head"]   = compute_quaternion(def_dir_head,   head - neck)

            # --- Arms / Legs / ToeBase ---
            for bone, idx in MIXAMO_MAP.items():
                parent = pts[idx]
                child = pts[idx + 2] if idx + 2 < len(pts) else pts[idx]

                if bone not in default_dirs:
                    # GLB 側に default_dir が無い場合はスキップ or 単位回転
                    frame_data[bone] = [0, 0, 0, 1]
                    continue

                default_dir = np.array(default_dirs[bone], dtype=float)
                target_dir = child - parent

                q = compute_quaternion(default_dir, target_dir)
                frame_data[bone] = q

            anim["frames"].append(frame_data)

        return anim

    converted = convert_to_mixamo_json(seq["frames"])
    payload = json.dumps(converted)

    # GLB を base64 で Three.js に渡す
    model_data = base64.b64encode(model_path.read_bytes()).decode()

    # 動画と Three.js を同じ HTML 内に統合
    html_code = """
    <div style="display:flex; flex-direction:column; gap:20px;">
      <div style="flex:1;">
        <h3>オリジナル動画</h3>
        <video id="video" width="100%" controls>
          <source src="VIDEO_PLACEHOLDER" type="video/mp4">
        </video>
      </div>
      <div style="flex:1;">
        <h3>3Dアバター再生</h3>
        <div id="container" style="width:100%; height:600px;"></div>
      </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/controls/OrbitControls.js"></script>
    <script>
    let avatar;

    // --- Three.js 基本セットアップ ---
    const container = document.getElementById('container');
    const w = container.clientWidth || window.innerWidth;
    const h = container.clientHeight || 600;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const light = new THREE.HemisphereLight(0xffffff, 0x444444, 1.0);
    light.position.set(0, 20, 0);
    scene.add(light);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5, 10, 7.5);
    scene.add(dirLight);

    const camera = new THREE.PerspectiveCamera(60, w/h, 0.01, 1000);
    camera.position.set(0, 1.5, 3);
    camera.lookAt(0, 1.5, 0);

    const renderer = new THREE.WebGLRenderer({antialias:true});
    renderer.setSize(w, h);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1.5, 0);
    controls.update();

    const axesHelper = new THREE.AxesHelper(1.0);
    scene.add(axesHelper);
    const gridHelper = new THREE.GridHelper(4, 20);
    scene.add(gridHelper);

    // --- GLB 読み込み ---
    const loader = new THREE.GLTFLoader();
    loader.load("data:application/octet-stream;base64,MODEL_PLACEHOLDER", function(gltf){
      scene.add(gltf.scene);
      avatar = gltf.scene;

      avatar.traverse(node => {
          console.log(node.name);
      });

      avatar.updateMatrixWorld(true);
    });

    // --- 動画同期 ---
    const video = document.getElementById("video");
    video.addEventListener("play", () => tick());
    video.addEventListener("seeked", () => tick());
    video.addEventListener("timeupdate", () => tick());

    // --- JSON アニメーション再生 ---
    const anim = PAYLOAD_PLACEHOLDER;

    function tick(){
      requestAnimationFrame(tick);
      if (!video || video.paused || !avatar) return;

      const frameIndex = Math.floor(video.currentTime * anim.frames.length / video.duration);
      const frame = anim.frames[Math.min(frameIndex, anim.frames.length - 1)];

      // Hips の位置
      const p = frame["mixamorig:Hips_pos"];
      avatar.position.set(p[0], p[1], p[2]);

      // 各ボーンの回転
      for (const boneName in frame) {
          if (!boneName.startsWith("mixamorig:")) continue;
          const bone = avatar.getObjectByName(boneName);
          if (!bone) {
            console.warn("Bone not found:", boneName);
            continue;
          }
          const q = frame[boneName];
          bone.quaternion.set(q[0], q[1], q[2], q[3]);
      }

      controls.update();
      renderer.render(scene, camera);
    }
    </script>
    """
    html_code = html_code.replace("PAYLOAD_PLACEHOLDER", payload)
    html_code = html_code.replace("MODEL_PLACEHOLDER", model_data)
    html_code = html_code.replace(
        "VIDEO_PLACEHOLDER",
        f"data:video/mp4;base64,{base64.b64encode(open(tmp_path,'rb').read()).decode()}"
    )
    st.components.v1.html(html_code, height=700, scrolling=False)
