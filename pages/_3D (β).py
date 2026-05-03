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


# 二重表示を防ぐためのコンテナ
if "processed" not in st.session_state:
    st.session_state.processed = False

uploaded = st.file_uploader("Upload your video", type=["mp4", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    with st.spinner("MediaPipe Analyzing..."):
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
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_tracker.process(cv2.resize(rgb, (448, 448)))

            if results.pose_landmarks:
                frames_data.append([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            else:
                frames_data.append(frames_data[-1] if frames_data else [[0,0,0]]*33)
        cap.release()

    video_b64 = base64.b64encode(open(video_path, "rb").read()).decode()
    
    # モデルの取得（キャッシュ推奨ですが一旦そのまま）
    vrm_url = "https://raw.githubusercontent.com/Keene23aZ4/ski-analyzer/main/pages/model.vrm"
    vrm_b64 = base64.b64encode(requests.get(vrm_url).content).decode()
    payload = json.dumps({"fps": fps, "frames": frames_data})

    # JS/HTML
    # ポイント：f-stringを使わず、波括弧エラーを回避
    html_template = """
    <div id="root" style="width:100%; display:flex; flex-direction:column; align-items:center;">
        <video id="v" width="100%" controls style="border-radius:10px; background:#000;">
            <source src="data:video/mp4;base64,VAR_VIDEO_B64">
        </video>
        <div id="c" style="width:100%; height:500px; background:#222; margin-top:10px; border-radius:10px;"></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/kalidokit@1.1.0/dist/kalidokit.umd.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@1.0.11/lib/three-vrm.js"></script>

    <script>
    (function() {
        const animData = VAR_PAYLOAD;
        const container = document.getElementById('c');
        const video = document.getElementById('v');

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x333333);
        
        const camera = new THREE.PerspectiveCamera(45, container.clientWidth/500, 0.1, 100);
        camera.position.set(0, 1.2, 3);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, 500);
        renderer.outputEncoding = THREE.sRGBEncoding;
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 1, 0);

        scene.add(new THREE.GridHelper(10, 20));
        scene.add(new THREE.AmbientLight(0xffffff, 0.7));
        const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
        dirLight.position.set(1, 2, 3);
        scene.add(dirLight);

        // --- デバッグ用：赤い箱（これが見えれば3D描画自体は成功） ---
        const box = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.2, 0.2), new THREE.MeshStandardMaterial({color: 0xff0000}));
        box.position.set(0, 0.1, 0);
        scene.add(box);

        let currentVRM = null;

        // VRM読み込み
        const loader = new THREE.GLTFLoader();
        const vrmData = "VAR_VRM_B64";
        
        const binary = atob(vrmData);
        const buf = new Uint8Array(binary.length);
        for (let i=0; i<binary.length; i++) buf[i] = binary.charCodeAt(i);

        loader.parse(buf.buffer, "", (gltf) => {
            // three-vrmの読み込み待ちを確実にする
            const vrmLib = window.threevrm || (THREE ? THREE.VRM : null);
            if (!vrmLib) {
                console.error("VRM Library not found");
                return;
            }
            vrmLib.VRM.from(gltf).then((vrm) => {
                currentVRM = vrm;
                scene.add(vrm.scene);
                vrm.scene.rotation.y = Math.PI;
                // 読み込み成功したら赤い箱を消す
                scene.remove(box);
                console.log("VRM Loaded");
            });
        }, (err) => console.error("Loader Error:", err));

        function animate() {
            requestAnimationFrame(animate);
            if (currentVRM && animData.frames.length) {
                let f = Math.floor(video.currentTime * animData.fps);
                if (f >= animData.frames.length) f = animData.frames.length - 1;
                const pts = animData.frames[f];
                
                const mpLandmarks = pts.map(p => ({
                    x: p[0], y: 1-p[1], z: -p[2], visibility: 1
                }));

                const pose = Kalidokit.Pose.solve(mpLandmarks, { runtime: "mediapipe" });
                if (pose) Kalidokit.VRMUtils.animateVRM(currentVRM, pose);
            }
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
    
    # 二重表示を避けるため、一回だけ描画する工夫
    st.components.v1.html(html_code, height=1100, scrolling=False)
