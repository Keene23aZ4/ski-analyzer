import streamlit as st
import cv2
import numpy as np
import json
import tempfile
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


# MediaPipe読み込み
try:
    import mediapipe as mp
    from mediapipe.python.solutions import pose as mp_pose
    Pose = mp_pose.Pose
except:
    import mediapipe.solutions.pose as mp_pose
    Pose = mp_pose.Pose

# --- Page Setup ---
st.set_page_config(page_title="3D Plot Avatar", layout="centered")
st.title("3D Motion Analysis (β)")
uploaded = st.file_uploader("Upload your video", type=["mp4", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    with st.spinner("MODELING..."):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        pose_tracker = Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
        
        frames_data = []
        prev_pts = None  # ← 正しい位置
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
        
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_tracker.process(rgb)
        
            if results.pose_world_landmarks:
                lm = results.pose_world_landmarks.landmark
                frame_pts = []
        
                for i in range(33):
                    p = lm[i]
                    if p.visibility < 0.5:
                        frame_pts.append(None)
                    else:
                        frame_pts.append([p.x, -p.y, -p.z])
        
                # 欠損補完
                if prev_pts is not None:
                    for i in range(33):
                        if frame_pts[i] is None:
                            frame_pts[i] = prev_pts[i]
        
                prev_pts = frame_pts
                frames_data.append(frame_pts)
        
            else:
                if prev_pts is not None:
                    frames_data.append(prev_pts)
                else:
                    frames_data.append([[0,0,0]] * 33)

    cap.release()
    pose_tracker.close()
    
    video_bytes = open(video_path, "rb").read()
    video_b64 = base64.b64encode(video_bytes).decode()
    import requests
    
    vrm_url = "https://raw.githubusercontent.com/Keene23aZ4/ski-analyzer/main/pages/model.vrm"
    vrm_bytes = requests.get(vrm_url).content
    vrm_b64 = base64.b64encode(vrm_bytes).decode()
    payload = json.dumps({"fps": fps, "frames": frames_data})
    
    # 修正ポイント: html_codeの定義から末尾までインデントを正確に揃えました
    html_code = f"""
    <div style="display: flex; flex-direction: column; align-items: center; gap: 15px;">
        <video id="sync_video" width="100%" controls playsinline style="border-radius: 12px; border: 1px solid #ccc;">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        <div id="container" style="width:100%; height:600px; background:#ffffff; border-radius:12px; overflow:hidden; border: 1px solid #eaeaea;"></div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@1.0.11/lib/three-vrm.js"></script>

    
    <script>
        const video = document.getElementById('sync_video');
        const container = document.getElementById('container');
        const animData = {payload};
    
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1c2833);
    
        const camera = new THREE.PerspectiveCamera(40, container.clientWidth/600, 0.1, 100);
        camera.position.set(6, 4, 8);
    
        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(container.clientWidth, 600);
        renderer.shadowMap.enabled = true;
        container.appendChild(renderer.domElement);
    
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
    
        scene.add(new THREE.GridHelper(10, 20, 0x0088ff, 0xdddddd));
        scene.add(new THREE.AxesHelper(5));
    
        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const light = new THREE.DirectionalLight(0xffffff, 0.7);
        light.position.set(5, 10, 5);
        light.castShadow = true;
        scene.add(light);

        
        // ===== VRM ローダー（Web版対応） =====
        let currentVRM = null;
        const loader = new THREE.GLTFLoader();
        
        const vrmBase64 = "{vrm_b64}";
        
        function base64ToArrayBuffer(base64) {{
            const binary = atob(base64);
            const len = binary.length;
            const buffer = new ArrayBuffer(len);
            const view = new Uint8Array(buffer);
            for (let i = 0; i < len; i++) {{
                view[i] = binary.charCodeAt(i);
            }}
            return buffer;
        }}
        
        const vrmBuffer = base64ToArrayBuffer(vrmBase64);
        
        loader.parse(
            vrmBuffer,
            "",
            (gltf) => {{
                THREE.VRM.from(gltf).then((vrm) => {{
                    currentVRM = vrm;
                    scene.add(vrm.scene);
        
                    vrm.scene.rotation.y = Math.PI;
                    vrm.scene.scale.set(0.1, 0.1, 0.1);
                }});
            }},
            (err) => {{
                console.error("VRM parse error:", err);
            }}
        );

    
        // ===== VRM アニメーション =====
        // 前フレームの landmark を保持
        let prevLandmarks = null;
        
        function updateAvatar() {{
            if (!animData.frames.length) return;
        
            let fIdx = Math.floor(video.currentTime * animData.fps);
            if (fIdx >= animData.frames.length) fIdx = animData.frames.length - 1;
        
            const raw = animData.frames[fIdx];
        
            if (!raw || !Array.isArray(raw) || raw.length !== 33) return;
 
            const mpLandmarks = raw.map((p, i) => {{
                if (!p) {{
                    return prevLandmarks ? prevLandmarks[i] : {{ x: 0, y: 0, z: 0 }};
                }}
                return {{ x: p[0], y: p[1], z: p[2] }};
            }});
        
            prevLandmarks = mpLandmarks;
        
            const kalidoPose = Kalidokit.Pose.solve(mpLandmarks, {{
                runtime: "mediapipe",
            }});
        
            if (currentVRM) {{
                Kalidokit.VRMUtils.animateVRM(currentVRM, kalidoPose);
            }}
        }}

    
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            updateAvatar();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
    """
    
    st.components.v1.html(html_code, height=1250)
