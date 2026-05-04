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
    <div id="root" style="width:100%; display:flex; flex-direction:column; align-items:center; font-family: sans-serif;">
        <video id="v" width="100%" controls style="border-radius:10px; background:#000;">
            <source src="data:video/mp4;base64,VAR_VIDEO_B64">
        </video>
        <div id="c" style="width:100%; height:500px; background:#1c2833; margin-top:10px; border-radius:10px; position:relative;">
            <!-- 画面上に状態を表示するインジケーター -->
            <div id="status" style="position:absolute; top:10px; left:10px; color:white; background:rgba(0,0,0,0.5); padding:5px 10px; border-radius:5px; font-size:12px; z-index:100;">
                Initializing...
            </div>
        </div>
    </div>

    <!-- ライブラリ読み込み -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/kalidokit@1.1.0/dist/kalidokit.umd.js"></script>
    <!-- unpkg版を使用（広告ブロックに強い傾向があります） -->
    <script src="https://unpkg.com/@pixiv/three-vrm@1.0.11/lib/three-vrm.js"></script>

    <script>
    (function() {
        const animData = VAR_PAYLOAD;
        const container = document.getElementById('c');
        const video = document.getElementById('v');
        const statusEl = document.getElementById('status');
        let currentVRM = null;

        // --- Three.js 基本設定 ---
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
        scene.add(new THREE.GridHelper(10, 20));
        scene.add(new THREE.AmbientLight(0xffffff, 0.7));
        const light = new THREE.DirectionalLight(0xffffff, 1.0);
        light.position.set(1, 2, 3);
        scene.add(light);

        // --- ライブラリを探すための強力なロジック ---
        let loadAttempts = 0;
        function initVRM() {
            loadAttempts++;
            statusEl.innerText = `Loading VRM Library (Attempt ${loadAttempts})...`;

            // 1. 複数の名前空間をチェック
            const vrmLib = window.threevrm || (THREE ? THREE.VRM : null);
            
            if (!vrmLib) {
                if (loadAttempts > 50) {
                    statusEl.style.color = "red";
                    statusEl.innerText = "Error: VRM Library Blocked. Please disable AdBlock.";
                    return;
                }
                setTimeout(initVRM, 200);
                return;
            }

            statusEl.innerText = "Library Found. Loading Model...";
            
            const loader = new THREE.GLTFLoader();
            try {
                const binary = atob("VAR_VRM_B64");
                const buf = new Uint8Array(binary.length);
                for (let i=0; i<binary.length; i++) buf[i] = binary.charCodeAt(i);

                loader.parse(buf.buffer, "", (gltf) => {
                    // vrmLibが直接 from を持っているか、.VRM 下にあるか
                    const factory = vrmLib.from ? vrmLib : vrmLib.VRM;
                    if (!factory || !factory.from) {
                        console.error("Factory not found in", vrmLib);
                        return;
                    }

                    factory.from(gltf).then((vrm) => {
                        currentVRM = vrm;
                        scene.add(vrm.scene);
                        vrm.scene.rotation.y = Math.PI;
                        statusEl.innerText = "✅ Ready";
                        setTimeout(() => statusEl.style.display = "none", 2000);
                    }).catch(e => {
                        console.error(e);
                        statusEl.innerText = "Model Initialization Error";
                    });
                });
            } catch (e) {
                console.error(e);
                statusEl.innerText = "Base64 Decode Error";
            }
        }
        initVRM();

        // --- アニメーションループ ---
        function updateAvatar() {
            if (!currentVRM || !animData || !animData.frames) return;
            
            let f = Math.floor(video.currentTime * animData.fps);
            if (f >= animData.frames.length) f = animData.frames.length - 1;
            const pts = animData.frames[f];
            
            if (!pts || pts.length < 33) return;

            // データがundefinedでも止まらないようにガード
            const mpLandmarks = pts.map(p => ({
                x: p ? (p[0] || 0) : 0,
                y: p ? (1 - (p[1] || 0)) : 0,
                z: p ? (-(p[2] || 0)) : 0,
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
