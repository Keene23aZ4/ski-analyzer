import streamlit as st
import cv2
import json
import tempfile
import mediapipe as mp
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

st.set_page_config(page_title="3D Pose → Avatar Motion", page_icon="🕺", layout="wide")

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
        if not ret: break
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

uploaded = st.file_uploader("動画をアップロード", type=["mp4","mov","avi","mkv"])
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpf:
        tmpf.write(uploaded.read())
        tmp_path = tmpf.name

    seq = extract_3d_pose_sequence(tmp_path, stride=3)
    payload = json.dumps({
        "frames": seq["frames"],
        "names": seq["landmark_names"],
        "fps": max(10.0, min(seq["fps"], 60.0)),
    })
    model_path = Path("static/avatar.glb")
    model_data = base64.b64encode(model_path.read_bytes()).decode()
    
    # 動画と Three.js を同じ HTML 内に統合
    html_code = """
    <div style="display:flex; gap:20px;">
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
    
    let hips, spine, neck;
    let leftUpperArm, leftForeArm, leftHand;
    let rightUpperArm, rightForeArm, rightHand;
    let leftUpLeg, leftLeg, leftFoot;
    let rightUpLeg, rightLeg, rightFoot;
    
    // --- MediaPipe payload ---
    const payload = PAYLOAD_PLACEHOLDER;
    
    // --- Mixamo 初期方向ベクトル ---
    const defaultDirs = {};
    
    function saveDefaultDir(bone, childBone) {
      const p = new THREE.Vector3();
      const c = new THREE.Vector3();
      bone.getWorldPosition(p);
      childBone.getWorldPosition(c);
      defaultDirs[bone.name] = c.clone().sub(p).normalize();
    }
    
    function rotateBone(bone, defaultDir, parentPos, childPos) {
      const targetDir = childPos.clone().sub(parentPos).normalize();
      const q = new THREE.Quaternion().setFromUnitVectors(defaultDir, targetDir);
      bone.quaternion.copy(q);
    }
    
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
    
    // --- GLB 読み込み ---
    const loader = new THREE.GLTFLoader();
    loader.load("data:application/octet-stream;base64,MODEL_PLACEHOLDER", function(gltf){
      scene.add(gltf.scene);
      avatar = gltf.scene;
    
      hips = avatar.getObjectByName("mixamorigHips");
      spine = avatar.getObjectByName("mixamorigSpine2");
      neck = avatar.getObjectByName("mixamorigNeck");
    
      leftUpperArm = avatar.getObjectByName("mixamorigLeftArm");
      leftForeArm = avatar.getObjectByName("mixamorigLeftForeArm");
      leftHand = avatar.getObjectByName("mixamorigLeftHand");
    
      rightUpperArm = avatar.getObjectByName("mixamorigRightArm");
      rightForeArm = avatar.getObjectByName("mixamorigRightForeArm");
      rightHand = avatar.getObjectByName("mixamorigRightHand");
    
      leftUpLeg = avatar.getObjectByName("mixamorigLeftUpLeg");
      leftLeg = avatar.getObjectByName("mixamorigLeftLeg");
      leftFoot = avatar.getObjectByName("mixamorigLeftFoot");
    
      rightUpLeg = avatar.getObjectByName("mixamorigRightUpLeg");
      rightLeg = avatar.getObjectByName("mixamorigRightLeg");
      rightFoot = avatar.getObjectByName("mixamorigRightFoot");
    
      avatar.updateMatrixWorld(true);
    
      saveDefaultDir(leftUpLeg, leftLeg);
      saveDefaultDir(leftLeg, leftFoot);
      saveDefaultDir(leftFoot, avatar.getObjectByName("mixamorigLeftToeBase"));   // ★追加
      saveDefaultDir(rightUpLeg, rightLeg);
      saveDefaultDir(rightLeg, rightFoot);
      saveDefaultDir(rightFoot, avatar.getObjectByName("mixamorigRightToeBase")); // ★追加
      saveDefaultDir(hips, spine);
      saveDefaultDir(spine, neck);
      console.log("defaultDirs:", defaultDirs);
      console.log("leftFoot:", leftFoot);
      console.log("rightFoot:", rightFoot);
      avatar.traverse(node => {
          if (node.name.toLowerCase().includes("foot"))
              console.log("FOOT FOUND:", node.name);
    });
    });
    
    // --- 動画同期 ---
    const video = document.getElementById("video");
    video.addEventListener("play", () => tick());
    video.addEventListener("seeked", () => tick());
    video.addEventListener("timeupdate", () => tick());
    
    // --- メインループ ---
    function tick(){
      requestAnimationFrame(tick);
      if (!video || video.paused || !avatar) return;
    
      const frameIndex = Math.floor(video.currentTime * payload.fps) % payload.frames.length;
      const LM = payload.frames[frameIndex].landmarks;
    
      const v = (i) => new THREE.Vector3(LM[i].x, -LM[i].y, -LM[i].z);
    
      // 腕
      rotateBone(leftUpperArm, defaultDirs["mixamorigLeftArm"], v(11), v(13));
      rotateBone(leftForeArm, defaultDirs["mixamorigLeftForeArm"], v(13), v(15));
    
      rotateBone(rightUpperArm, defaultDirs["mixamorigRightArm"], v(12), v(14));
      rotateBone(rightForeArm, defaultDirs["mixamorigRightForeArm"], v(14), v(16));
    
      // 脚
      rotateBone(leftUpLeg, defaultDirs["mixamorigLeftUpLeg"], v(23), v(25));
      rotateBone(leftLeg, defaultDirs["mixamorigLeftLeg"], v(25), v(27));
      rotateBone(leftFoot, defaultDirs["mixamorigLeftFoot"], v(27), v(31));
    
      rotateBone(rightUpLeg, defaultDirs["mixamorigRightUpLeg"], v(24), v(26));
      rotateBone(rightLeg, defaultDirs["mixamorigRightLeg"], v(26), v(28));
      rotateBone(rightFoot, defaultDirs["mixamorigRightFoot"], v(28), v(32));
    
      // 胴体
      rotateBone(hips, defaultDirs["mixamorigHips"], v(23), v(11));
      rotateBone(spine, defaultDirs["mixamorigSpine2"], v(11), v(0));
    
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
