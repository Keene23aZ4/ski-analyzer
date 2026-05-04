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



uploaded = st.file_uploader("Upload your video", type=["mp4", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    with st.spinner("Analyzing Pose..."):
        import mediapipe as mp
        from mediapipe.python.solutions import pose as mp_pose
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        pose_tracker = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        frames_data = []
        last_valid_frame = [[0, 0, 0]] * 33
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_tracker.process(cv2.resize(rgb, (320, 320)))

            if results.pose_landmarks:
                current_frame = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
                frames_data.append(current_frame)
                last_valid_frame = current_frame
            else:
                frames_data.append(last_valid_frame)
        cap.release()

    video_b64 = base64.b64encode(open(video_path, "rb").read()).decode()
    vrm_url = "https://raw.githubusercontent.com/Keene23aZ4/ski-analyzer/main/pages/model.vrm"
    vrm_b64 = base64.b64encode(requests.get(vrm_url).content).decode()
    payload = json.dumps({"fps": fps, "frames": frames_data})

    html_template = """
    <div id="root" style="width:100%; display:flex; flex-direction:column; align-items:center;">
        <video id="v" width="100%" controls style="border-radius:10px; background:#000;">
            <source src="data:video/mp4;base64,VAR_VIDEO_B64">
        </video>
        <div id="c" style="width:100%; height:500px; background:#1c2833; margin-top:10px; border-radius:10px;"></div>
    </div>

    <!-- ライブラリ読み込み: 順番とソースを微調整 -->
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/kalidokit@1.1.0/dist/kalidokit.umd.js"></script>
    <!-- VRMライブラリ: window.threevrm を確実に作るビルドを指定 -->
    <script src="https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@1.0.11/lib/three-vrm.js"></script>

    <script>
    (function() {
        const animData = VAR_PAYLOAD;
        const container = document.getElementById('c');
        const video = document.getElementById('v');
        let currentVRM = null;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1c2833);
        const camera = new THREE.PerspectiveCamera(40, container.clientWidth/500, 0.1, 100);
        camera.position.set(0, 1.4, 3.5);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, 500);
        renderer.outputEncoding = THREE.sRGBEncoding;
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 1.0, 0);
        scene.add(new THREE.GridHelper(10, 20, 0x444444, 0x222222));
        scene.add(new THREE.AmbientLight(0xffffff, 0.8));
        const light = new THREE.DirectionalLight(0xffffff, 1.0);
        light.position.set(1, 2, 3);
        scene.add(light);

        // --- ライブラリの存在確認を強化 ---
        function getVRMLib() {
            // window.threevrm または THREE.VRM を探す
            if (typeof threevrm !== 'undefined') return threevrm;
            if (THREE && THREE.VRM) return THREE;
            return null;
        }

        function loadVRM() {
            const vrmLib = getVRMLib();
            if (!vrmLib) {
                console.log("Still waiting for VRM Library...");
                setTimeout(loadVRM, 500); // 待機時間を少し伸ばして再試行
                return;
            }

            const loader = new THREE.GLTFLoader();
            const binary = atob("VAR_VRM_B64");
            const buf = new Uint8Array(binary.length);
            for (let i=0; i<binary.length; i++) buf[i] = binary.charCodeAt(i);

            loader.parse(buf.buffer, "", (gltf) => {
                // vrmLib自体に .VRM があるか、vrmLib.VRM に .from があるか
                const factory = vrmLib.VRM || vrmLib;
                factory.from(gltf).then((vrm) => {
                    currentVRM = vrm;
                    scene.add(vrm.scene);
                    vrm.scene.rotation.y = Math.PI;
                    console.log("✅ VRM Model Loaded!");
                }).catch(e => console.error("VRM Factory Error:", e));
            }, (err) => console.error("GLTF Parse Error:", err));
        }
        loadVRM();

        function updateAvatar() {
            if (!currentVRM || !animData.frames.length) return;
            
            let f = Math.floor(video.currentTime * animData.fps);
            if (f >= animData.frames.length) f = animData.frames.length - 1;
            const pts = animData.frames[f];
            
            if (!pts || pts.length < 33) return;

            const mpLandmarks = pts.map(p => ({
                x: p[0] || 0,
                y: 1 - (p[1] || 0),
                z: -(p[2] || 0),
                visibility: 1
            }));

            try {
                const pose = Kalidokit.Pose.solve(mpLandmarks, { runtime: "mediapipe" });
                if (pose) {
                    Kalidokit.VRMUtils.animateVRM(currentVRM, pose);
                }
            } catch (err) {}
        }

        function animate() {
            requestAnimationFrame(animate);
            updateAvatar();
            controls.update();
            renderer.render(scene, camera);
        }
        animate();
    })();
    </script>
    """

    html_code = html_template.replace("VAR_VIDEO_B64", video_b64)\
                             .replace("VAR_VRM_B64", vrm_b64)\
                             .replace("VAR_PAYLOAD", payload)
    
    st.components.v1.html(html_code, height=1100, scrolling=False)
