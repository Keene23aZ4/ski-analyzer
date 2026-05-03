

import streamlit as st
import cv2
import numpy as np
import json
import tempfile
import base64
from pathlib import Path
import requests

# --- Page Setup ---
st.set_page_config(page_title="3D Plot Avatar", layout="centered")
st.title("3D Motion Analysis (β)")
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


# 表示用プレースホルダー（二重表示防止）
container_placeholder = st.empty()

uploaded = st.file_uploader("Upload your video", type=["mp4", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    with st.spinner("Processing Video..."):
        # MediaPipe読み込み
        import mediapipe as mp
        from mediapipe.python.solutions import pose as mp_pose
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        pose_tracker = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        frames_data = []
        prev_pts = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 解像度を少し上げる（精度向上のため）
            rgb_small = cv2.resize(rgb, (480, 480))
            results = pose_tracker.process(rgb_small)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                frame_pts = []
                for i in range(33):
                    p = lm[i]
                    # 信頼度が低い場合は前フレームを使用
                    if p.visibility < 0.5 and prev_pts is not None:
                        frame_pts.append(prev_pts[i])
                    else:
                        frame_pts.append([p.x, p.y, p.z])
                prev_pts = frame_pts
                frames_data.append(frame_pts)
            else:
                frames_data.append(prev_pts if prev_pts else [[0.5, 0.5, 0]] * 33)

    cap.release()
    pose_tracker.close()
    
    # データの準備
    video_bytes = open(video_path, "rb").read()
    video_b64 = base64.b64encode(video_bytes).decode()
    
    vrm_url = "https://raw.githubusercontent.com/Keene23aZ4/ski-analyzer/main/pages/model.vrm"
    vrm_bytes = requests.get(vrm_url).content
    vrm_b64 = base64.b64encode(vrm_bytes).decode()
    payload = json.dumps({"fps": fps, "frames": frames_data})
    
    # JavaScript/HTML テンプレート
    html_template = """
    <div id="wrapper" style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
        <video id="sync_video" width="100%" controls playsinline style="border-radius: 8px; border: 1px solid #444;">
            <source src="data:video/mp4;base64,VAR_VIDEO_B64" type="video/mp4">
        </video>
        <div id="three_container" style="width:100%; height:500px; background:#1c2833; border-radius:8px; position:relative;"></div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/kalidokit@1.1.0/dist/kalidokit.umd.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@1.0.11/lib/three-vrm.js"></script>

    <script>
        const video = document.getElementById('sync_video');
        const container = document.getElementById('three_container');
        const animData = VAR_PAYLOAD;
    
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, container.clientWidth/500, 0.1, 100);
        camera.position.set(0, 1.2, 3); // 人物の正面位置
    
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(container.clientWidth, 500);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.outputEncoding = THREE.sRGBEncoding;
        container.appendChild(renderer.domEncoding ? renderer.domElement : renderer.domElement);
    
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 1, 0);
        controls.update();
    
        scene.add(new THREE.GridHelper(10, 20));
        scene.add(new THREE.AmbientLight(0xffffff, 0.8));
        const light = new THREE.DirectionalLight(0xffffff, 1.0);
        light.position.set(1, 2, 1);
        scene.add(light);

        let currentVRM = null;
        const loader = new THREE.GLTFLoader();
        
        function base64ToArrayBuffer(base64) {
            const binary = atob(base64);
            const buffer = new ArrayBuffer(binary.length);
            const view = new Uint8Array(buffer);
            for (let i = 0; i < binary.length; i++) view[i] = binary.charCodeAt(i);
            return buffer;
        }
        
        // VRMモデルのロード
        const vrmBuffer = base64ToArrayBuffer("VAR_VRM_B64");
        loader.parse(vrmBuffer, "", (gltf) => {
            // three-vrm 1.x 系の初期化
            const vrmLib = window.threevrm || THREE;
            vrmLib.VRM.from(gltf).then((vrm) => {
                currentVRM = vrm;
                scene.add(vrm.scene);
                vrm.scene.rotation.y = Math.PI; // 正面に向ける
                console.log("VRM Model Ready");
            });
        }, (err) => console.error("Loader Error:", err));

        function updateAvatar() {
            if (!currentVRM || !animData.frames.length) return;
        
            let fIdx = Math.floor(video.currentTime * animData.fps);
            if (fIdx >= animData.frames.length) fIdx = animData.frames.length - 1;
        
            const raw = animData.frames[fIdx];
            if (!raw) return;

            const mpLandmarks = raw.map(p => ({
                x: p[0],
                y: 1 - p[1],
                z: -p[2],
                visibility: 1
            }));

            const kalidoPose = Kalidokit.Pose.solve(mpLandmarks, { runtime: "mediapipe" });
            if (kalidoPose) {
                Kalidokit.VRMUtils.animateVRM(currentVRM, kalidoPose);
            }
        }

        function animate() {
            requestAnimationFrame(animate);
            updateAvatar();
            renderer.render(scene, camera);
        }
        animate();
    </script>
    """
    
    # 文字列置換
    html_code = html_template.replace("VAR_VIDEO_B64", video_b64)\
                             .replace("VAR_VRM_B64", vrm_b64)\
                             .replace("VAR_PAYLOAD", payload)
    
    # 出力（placeholderを使うことで二重描画を防ぐ）
    with container_placeholder:
        st.components.v1.html(html_code, height=1100)
