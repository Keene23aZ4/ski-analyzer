import streamlit as st
import cv2
import numpy as np
import json
import tempfile
import mediapipe as mp
import base64
from pathlib import Path

from retarget import extract_default_dirs  # GLB から default_dir を取る

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

    # ✅ GLB から Tポーズの default_dir を取得（コロン無しに変換）
    model_path = Path("static/avatar.glb")
    default_dirs_raw = extract_default_dirs(str(model_path))
    default_dirs = {k.replace("mixamorig:", "mixamorig"): v for k, v in default_dirs_raw.items()}

    # ✅ MediaPipe → Mixamo の対応（コロン無し）
    MIXAMO_MAP = {
        "mixamorigHips": 23,
        "mixamorigLeftArm": 11,
        "mixamorigLeftForeArm": 13,
        "mixamorigLeftHand": 15,
        "mixamorigRightArm": 12,
        "mixamorigRightForeArm": 14,
        "mixamorigRightHand": 16,
        "mixamorigLeftUpLeg": 23,
        "mixamorigLeftLeg": 25,
        "mixamorigLeftFoot": 27,
        "mixamorigLeftToeBase": 31,
        "mixamorigRightUpLeg": 24,
        "mixamorigRightLeg": 26,
        "mixamorigRightFoot": 28,
        "mixamorigRightToeBase": 32,
    }

    def mp_to_mixamo_vec(lm):
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

            hips = pts[23]
            neck = pts[0]
            head = pts[0]

            spine, spine1, spine2 = compute_spine_points(hips, neck)

            frame_data = {}
            frame_data["mixamorigHips_pos"] = hips.tolist()

            # ✅ Spine 系（コロン無し）
            for bone, vec in [
                ("mixamorigSpine", spine - hips),
                ("mixamorigSpine1", spine1 - spine),
                ("mixamorigSpine2", spine2 - spine1),
                ("mixamorigNeck", neck - spine2),
                ("mixamorigHead", head - neck),
            ]:
                default_dir = np.array(default_dirs.get(bone, [0, 1, 0]), dtype=float)
                frame_data[bone] = compute_quaternion(default_dir, vec)

            # ✅ Arms / Legs / ToeBase
            for bone, idx in MIXAMO_MAP.items():
                parent = pts[idx]
                child = pts[idx + 2] if idx + 2 < len(pts) else pts[idx]

                if bone not in default_dirs:
                    frame_data[bone] = [0, 0, 0, 1]
                    continue

                default_dir = np.array(default_dirs[bone], dtype=float)
                target_dir = child - parent

                frame_data[bone] = compute_quaternion(default_dir, target_dir)

            anim["frames"].append(frame_data)

        return anim

    converted = convert_to_mixamo_json(seq["frames"])
    payload = json.dumps(converted)

    # GLB を base64 で Three.js に渡す
    model_data = base64.b64encode(model_path.read_bytes()).decode()

    # ✅ Three.js 側もコロン無しで一致
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

    const loader = new THREE.GLTFLoader();
    loader.load("data:application/octet-stream;base64,MODEL_PLACEHOLDER", function(gltf){
      scene.add(gltf.scene);
      avatar = gltf.scene;
      avatar.updateMatrixWorld(true);
    });

    const video = document.getElementById("video");
    video.addEventListener("play", () => tick());
    video.addEventListener("seeked", () => tick());
    video.addEventListener("timeupdate", () => tick());

    const anim = PAYLOAD_PLACEHOLDER;

    function tick(){
      requestAnimationFrame(tick);
      if (!video || video.paused || !avatar) return;

      const frameIndex = Math.floor(video.currentTime * anim.frames.length / video.duration);
      const frame = anim.frames[Math.min(frameIndex, anim.frames.length - 1)];

      const p = frame["mixamorigHips_pos"];
      avatar.position.set(p[0], p[1], p[2]);

      for (const boneName in frame) {
          if (!boneName.startsWith("mixamorig")) continue;
          const bone = avatar.getObjectByName(boneName);
          if (!bone) continue;
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
