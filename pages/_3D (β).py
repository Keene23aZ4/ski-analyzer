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
    let leftShoulder, leftUpperArm, leftForeArm, leftHand;
    let rightShoulder, rightUpperArm, rightForeArm, rightHand;
    let leftUpLeg, leftLeg, leftFoot;
    let rightUpLeg, rightLeg, rightFoot;



    const payload = PAYLOAD_PLACEHOLDER;

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
    renderer.setSize(w,h);
    container.appendChild(renderer.domElement);
    window.addEventListener("load", () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      renderer.setSize(w, h);
    });
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1.5, 0);  // アバターの胸あたりを中心にする
    controls.update();


    // ★ アバターモデルを読み込む ★
    const loader = new THREE.GLTFLoader();
    loader.load("data:application/octet-stream;base64,MODEL_PLACEHOLDER", function(gltf){
      scene.add(gltf.scene);
      avatar = gltf.scene;
      console.log("=== GLB のノード一覧 ===");
      avatar.traverse(node => console.log(node.name));

      avatar.position.set(0, 0, 0);
      avatar.scale.set(1, 1, 1);
      avatar.rotation.set(0, 0, 0);

      console.log("avatar children:", avatar.children);
      avatar.traverse(function(node){
          if (node.isMesh){
          console.log("Mesh found:", node.name, node);
          }
      });
    
      hips = avatar.getObjectByName("mixamorigHips");
      spine = avatar.getObjectByName("mixamorigSpine2");
      neck = avatar.getObjectByName("mixamorigNeck");
      leftShoulder = avatar.getObjectByName("mixamorigLeftShoulder");
      leftUpperArm = avatar.getObjectByName("mixamorigLeftArm");
      leftForeArm = avatar.getObjectByName("mixamorigLeftForeArm");
      leftHand = avatar.getObjectByName("mixamorigLeftHand");
      rightShoulder = avatar.getObjectByName("mixamorigRightShoulder");
      rightUpperArm = avatar.getObjectByName("mixamorigRightArm");
      rightForeArm = avatar.getObjectByName("mixamorigRightForeArm");
      rightHand = avatar.getObjectByName("mixamorigRightHand");
      leftUpLeg = avatar.getObjectByName("mixamorigLeftUpLeg");
      leftLeg = avatar.getObjectByName("mixamorigLeftLeg");
      leftFoot = avatar.getObjectByName("mixamorigLeftFoot");
      rightUpLeg = avatar.getObjectByName("mixamorigRightUpLeg");
      rightLeg = avatar.getObjectByName("mixamorigRightLeg");
      rightFoot = avatar.getObjectByName("mixamorigRightFoot");
  });

    

    

    const video = document.getElementById("video");
    video.addEventListener("play", () => tick());
    video.addEventListener("seeked", () => tick());
    video.addEventListener("timeupdate", () => tick());
    function applyBoneRotation(bone, parentPos, childPos, baseDir, offsetQuat=null){
      const dir = childPos.clone().sub(parentPos).normalize();
      const q = new THREE.Quaternion().setFromUnitVectors(baseDir, dir);
    
      if (offsetQuat){
        q.multiply(offsetQuat);  // ← オフセットを適用
      }
    
      bone.quaternion.copy(q);
    }

    // Tポーズ → 直立姿勢のオフセット
    const offset = {
      arm: new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -Math.PI/2)),   // -90°
      forearm: new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -Math.PI/2)),
      leg: new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, Math.PI/12)),   // +15° 内側
      lowerLeg: new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, Math.PI/12)),
      spine: new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI/12, 0, 0)) // -15° 前傾
    };


    function tick(){
      requestAnimationFrame(tick);
      if (!video || video.paused) return;
    
      const frameIndex = Math.floor(video.currentTime * payload.fps) % payload.frames.length;
      const frame = payload.frames[frameIndex];
    
      
    
      // --- アバターの動き（全身） ---
      if (avatar){
          const LM = frame.landmarks;
          const v = (i) => new THREE.Vector3(LM[i].x, -LM[i].y, LM[i].z);
        
          // --- 胴体（+Z 基準） ---
          if (hips && spine){
            applyBoneRotation(hips, v(23), v(11), new THREE.Vector3(0, 0, 1));
          }
          if (spine && neck){
            applyBoneRotation(spine, v(11), v(0), new THREE.Vector3(0, 0, 1));
          }
        
          // --- 左腕（+X 基準） ---
          if (leftUpperArm){
            applyBoneRotation(leftUpperArm, v(11), v(13), new THREE.Vector3(1,0,0), offset.arm);
          }
          if (leftForeArm){
            applyBoneRotation(leftForeArm, v(13), v(15), new THREE.Vector3(1,0,0), offset.forearm);
          }
        
          // --- 右腕（+X 基準） ---
          if (rightUpperArm){
            applyBoneRotation(rightUpperArm, v(12), v(14), new THREE.Vector3(1,0,0), offset.arm);
          }
          if (rightForeArm){
            applyBoneRotation(rightForeArm, v(14), v(16), new THREE.Vector3(1,0,0), offset.forearm);
          }
        
          // --- 左脚（-Y 基準） ---
          if (leftUpLeg){
            applyBoneRotation(leftUpLeg, v(23), v(25), new THREE.Vector3(0, -1, 0));
          }
          if (leftLeg){
            applyBoneRotation(leftLeg, v(25), v(27), new THREE.Vector3(0, -1, 0));
          }
          if (leftFoot){
            applyBoneRotation(leftFoot, v(27), v(31), new THREE.Vector3(0, -1, 0));
          }
        
          // --- 右脚（-Y 基準） ---
          if (rightUpLeg){
            applyBoneRotation(rightUpLeg, v(24), v(26), new THREE.Vector3(0, -1, 0));
          }
          if (rightLeg){
            applyBoneRotation(rightLeg, v(26), v(28), new THREE.Vector3(0, -1, 0));
          }
          if (rightFoot){
            applyBoneRotation(rightFoot, v(28), v(32), new THREE.Vector3(0, -1, 0));
          }
        }
      controls.update();
      renderer.render(scene,camera);
    }
    tick();
    </script>
    """
    html_code = html_code.replace("PAYLOAD_PLACEHOLDER", payload)
    html_code = html_code.replace("MODEL_PLACEHOLDER", model_data)
    html_code = html_code.replace(
        "VIDEO_PLACEHOLDER",
        f"data:video/mp4;base64,{base64.b64encode(open(tmp_path,'rb').read()).decode()}"
    )
    st.components.v1.html(html_code, height=700, scrolling=False)
