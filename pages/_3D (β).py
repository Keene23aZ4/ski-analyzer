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

    # JS 部分は通常文字列にして、payload だけ置換
    three_js_code = """
    <div id="container" style="width:100%; height:600px;"></div>
    <script src="https://cdn.jsdelivr.net/npm/three@0.149.0/build/three.min.js"></script>
    <script>
    const payload = PAYLOAD_PLACEHOLDER;
    class OrbitControls extends THREE.EventDispatcher {
      constructor(object, domElement) {
        super();
        this.object = object;
        this.domElement = domElement;
        this.target = new THREE.Vector3();
        this.domElement.addEventListener('mousedown', (event) => {
          event.preventDefault();
          this.startX = event.clientX;
          this.startY = event.clientY;
          const onMove = (e) => {
            const dx = e.clientX - this.startX;
            const dy = e.clientY - this.startY;
            this.startX = e.clientX;
            this.startY = e.clientY;
            this.object.position.applyAxisAngle(new THREE.Vector3(0,1,0), dx*0.005);
            this.object.position.applyAxisAngle(new THREE.Vector3(1,0,0), dy*0.005);
            this.object.lookAt(this.target);
          };
          const onUp = () => {
            this.domElement.removeEventListener('mousemove', onMove);
            this.domElement.removeEventListener('mouseup', onUp);
          };
          this.domElement.addEventListener('mousemove', onMove);
          this.domElement.addEventListener('mouseup', onUp);
        });
      }
      update() { this.object.lookAt(this.target); }
    }
    THREE.OrbitControls = OrbitControls;



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

    // ランドマーク点
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

    // 動画タグを取得して currentTime に同期
    let video = null;
    
    function waitForVideo(){
      video = document.querySelector("video");
      if(video){
        console.log("Videoタグを取得しました:", video);
        tick();  // 動画が見つかったら tick 開始
      } else {
        console.log("Videoタグがまだ存在しません。再試行します...");
        setTimeout(waitForVideo, 500); // 0.5秒ごとにチェック
      }
    }
    
    function tick(){
      requestAnimationFrame(tick);
      if (!video || video.paused) return; // 動画が停止中なら骨格も止める
    
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
    
      renderer.render(scene,camera);
    }
    
    // 最初に動画タグを探す
    waitForVideo();
    </script>
    """

    html = three_js_code.replace("PAYLOAD_PLACEHOLDER", payload)

    # 動画と 3D ビューを並べて表示
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("オリジナル動画")
        st.video(tmp_path)

    with col2:
        st.subheader("3Dアバター（スティックフィギュア）再生")
        st.components.v1.html(html, height=700, scrolling=False)
