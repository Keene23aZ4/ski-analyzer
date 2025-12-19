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

# MediaPipe読み込み（エラー回避用）
try:
    import mediapipe as mp
    from mediapipe.python.solutions import pose as mp_pose
    Pose = mp_pose.Pose
except:
    import mediapipe.solutions.pose as mp_pose
    Pose = mp_pose.Pose


# --- Page Setup ---
st.set_page_config(page_title="3D Plot Avatar", layout="centered")
st.title("3D motion (β)")
uploaded = st.file_uploader("Upload your video", type=["mp4", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    with st.spinner("座標抽出中..."):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        pose_tracker = Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
        
        frames_data = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_tracker.process(rgb)
            if results.pose_world_landmarks:
                lm = results.pose_world_landmarks.landmark
                frame_pts = [[p.x, -p.y, -p.z] for p in lm]
                frames_data.append(frame_pts)
            else:
                frames_data.append(None)
        cap.release()
        pose_tracker.close()

    # 動画をHTMLに埋め込むためのBase64エンコード
    video_bytes = open(video_path, 'rb').read()
    video_b64 = base64.b64encode(video_bytes).decode()
    payload = json.dumps({"fps": fps, "frames": frames_data})

    # Three.js コード (動画同期版)
    html_code = f"""
    <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
        <video id="sync_video" width="100%" controls playsinline style="border-radius: 8px; border: 1px solid #ccc;">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        <div id="container" style="width:100%; height:500px; background:#ffffff; border:1px solid #ccc; border-radius:8px;"></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        const video = document.getElementById('sync_video');
        const container = document.getElementById('container');
        const animData = {payload}; 
        
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xfcfcfc); 
        
        const camera = new THREE.PerspectiveCamera(45, container.clientWidth/500, 0.1, 100);
        camera.position.set(4, 4, 6);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(container.clientWidth, 500);
        container.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 1, 0);
        controls.update();

        // 3Dグリッド
        scene.add(new THREE.GridHelper(10, 10, 0x888888, 0xdddddd));
        const gridXY = new THREE.GridHelper(10, 10, 0x888888, 0xeeeeee);
        gridXY.rotation.x = Math.PI / 2; gridXY.position.set(0, 5, -5); scene.add(gridXY);
        const gridYZ = new THREE.GridHelper(10, 10, 0x888888, 0xeeeeee);
        gridYZ.rotation.z = Math.PI / 2; gridYZ.position.set(-5, 5, 0); scene.add(gridYZ);

        scene.add(new THREE.AxesHelper(5));
        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
        dirLight.position.set(5, 10, 5); scene.add(dirLight);

        const skinMat = new THREE.MeshStandardMaterial({{ color: 0x3399ff, roughness: 0.5 }});
        const jointMat = new THREE.MeshStandardMaterial({{ color: 0x666666 }});
        const meshes = {{}};

        function createLimb(name, thickness) {{
            const geo = new THREE.CapsuleGeometry(thickness, 1, 4, 8);
            geo.rotateX(-Math.PI / 2); geo.translate(0, 0, 0.5);
            const mesh = new THREE.Mesh(geo, skinMat);
            scene.add(mesh); meshes[name] = mesh;
        }}
        
        function createJoint(i, r) {{
            const mesh = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 16), jointMat);
            scene.add(mesh); meshes['j' + i] = mesh;
        }}

        const conns = [
            [11,12,'sh',.05],[11,23,'ls',.06],[12,24,'rs',.06],[23,24,'hp',.07],
            [11,13,'la',.04],[13,15,'lf',.03],[12,14,'ra',.04],[14,16,'rf',.03],
            [23,25,'lt',.07],[25,27,'lsn',.05],[24,26,'rt',.07],[26,28,'rsn',.05]
        ];
        conns.forEach(c => createLimb(c[2], c[3]));
        [11,12,13,14,15,16,23,24,25,26,27,28,0].forEach(i => createJoint(i, 0.05));
        meshes['head'] = new THREE.Mesh(new THREE.SphereGeometry(0.14, 20, 20), skinMat);
        scene.add(meshes['head']);

        function updateAvatar() {{
            if (!animData.frames.length) return;

            // --- 重要: ビデオの現在の再生時間を基準にフレームを計算 ---
            let fIdx = Math.floor(video.currentTime * animData.fps);
            if (fIdx >= animData.frames.length) fIdx = animData.frames.length - 1;
            
            const raw = animData.frames[fIdx];
            if (!raw) return;

            const pts = raw.map(p => new THREE.Vector3(p[0]*4, p[1]*4 + 2.5, p[2]*4));

            for (let i=0; i<33; i++) {{
                if (meshes['j'+i]) meshes['j'+i].position.copy(pts[i]);
            }}
            if (meshes['head']) meshes['head'].position.copy(pts[0]);

            conns.forEach(c => {{
                const m = meshes[c[2]], pA = pts[c[0]], pB = pts[c[1]];
                if (m && pA && pB) {{
                    m.position.copy(pA);
                    m.lookAt(pB);
                    m.scale.set(1, 1, pA.distanceTo(pB));
                }}
            }});
        }}

        function animate() {{
            requestAnimationFrame(animate);
            updateAvatar();
            renderer.render(scene, camera);
        }}
        animate();

        window.addEventListener('resize', () => {{
            camera.aspect = container.clientWidth / 500;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, 500);
        }});
    </script>
    """
    st.components.v1.html(html_code, height=1050) # 動画+3Dの高さに合わせて調整
