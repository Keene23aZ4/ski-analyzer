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

    <script src="https://cdn.jsdelivr.net/npm/three@0.149.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.149.0/examples/js/loaders/GLTFLoader.js"></script>
    <script>
    const payload = PAYLOAD_PLACEHOLDER;

    const container = document.getElementById('container');
    const w = container.clientWidth || window.innerWidth;
    const h = container.clientHeight || 600;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const camera = new THREE.PerspectiveCamera(60, w/h, 0.01, 1000);
    camera.position.set(0,0,3);
    const renderer = new THREE.WebGLRenderer({antialias:true});
    renderer.setSize(w,h);
    container.appendChild(renderer.domElement);

    // ★ アバターモデルを読み込む ★
    const loader = new THREE.GLTFLoader();
    loader.load("data:model/gltf-binary;base64,MODEL_PLACEHOLDER", function(gltf){
      scene.add(gltf.scene);
      avatar = gltf.scene;
    
      leftShoulder = avatar.getObjectByName("LeftShoulder");
      rightShoulder = avatar.getObjectByName("RightShoulder");
      leftElbow = avatar.getObjectByName("LeftElbow");
      rightElbow = avatar.getObjectByName("RightElbow");
      leftHip = avatar.getObjectByName("LeftHip");
      rightHip = avatar.getObjectByName("RightHip");
    });

    // スティックフィギュア（ランドマーク点）
    const spheres = [];
    payload.names.forEach((name,i) => {
      const geom = new THREE.SphereGeometry(0.02,8,8);
      const mat = new THREE.MeshBasicMaterial({color:0x00ff00});
      const s = new THREE.Mesh(geom,mat);
      scene.add(s);
      spheres.push(s);
    });

    // 接続線
    const connections = [[11,13],[13,15],[12,14],[14,16],[11,12],[23,24],[23,25],[25,27],[24,26],[26,28]];
    const lineMaterial = new THREE.LineBasicMaterial({color:0xffffff});
    const lineGeometry = new THREE.BufferGeometry();
    const linePositions = new Float32Array(connections.length*2*3);
    lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions,3));
    const skeletonLines = new THREE.LineSegments(lineGeometry,lineMaterial);
    scene.add(skeletonLines);

    const video = document.getElementById("video");

    function tick(){
      requestAnimationFrame(tick);
      if (!video || video.paused) return;
    
      const frameIndex = Math.floor(video.currentTime * payload.fps) % payload.frames.length;
      const frame = payload.frames[frameIndex];
    
      frame.landmarks.forEach((lm,i)=>{
        spheres[i].position.set(lm.x,-lm.y,lm.z);
      });
    
      connections.forEach((conn,ci)=>{
        const a = frame.landmarks[conn[0]];
        const b = frame.landmarks[conn[1]];
        linePositions[ci*6+0]=a.x; linePositions[ci*6+1]=-a.y; linePositions[ci*6+2]=a.z;
        linePositions[ci*6+3]=b.x; linePositions[ci*6+4]=-b.y; linePositions[ci*6+5]=b.z;
      });
      skeletonLines.geometry.attributes.position.needsUpdate = true;
    
      // ★★★ アバターの動きを追加 ★★★
      if (avatar){
        if (leftShoulder){
          leftShoulder.position.set(frame.landmarks[11].x, -frame.landmarks[11].y, frame.landmarks[11].z);
        }
        if (rightShoulder){
          rightShoulder.position.set(frame.landmarks[12].x, -frame.landmarks[12].y, frame.landmarks[12].z);
        }
        if (leftElbow){
          leftElbow.position.set(frame.landmarks[13].x, -frame.landmarks[13].y, frame.landmarks[13].z);
        }
        if (rightElbow){
          rightElbow.position.set(frame.landmarks[14].x, -frame.landmarks[14].y, frame.landmarks[14].z);
        }
      }
    
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
