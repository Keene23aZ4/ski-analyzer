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

import streamlit as st

# MediaPipe読み込み
try:
    import mediapipe as mp
    from mediapipe.python.solutions import pose as mp_pose
    Pose = mp_pose.Pose
except:
    import mediapipe.solutions.pose as mp_pose
    Pose = mp_pose.Pose

st.set_page_config(page_title="Motion Visualize 3D (β)", layout="centered")
st.title("Motion Visualize 3D (β)")

uploaded = st.file_uploader("Upload your video!", type=["mp4", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    with st.spinner("ANALYSING..."):
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

    video_bytes = open(video_path, 'rb').read()
    video_b64 = base64.b64encode(video_bytes).decode()
    payload = json.dumps({"fps": fps, "frames": frames_data})

    # Three.js コード
    html_code = f"""
    <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
        <video id="sync_video" width="100%" controls playsinline style="border-radius: 8px;">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        <div id="container" style="width:100%; height:600px; background:#111; border-radius:8px;"></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        const video = document.getElementById('sync_video');
        const container = document.getElementById('container');
        const animData = {payload};
        
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x111111);
        
        const camera = new THREE.PerspectiveCamera(45, container.clientWidth/600, 0.1, 100);
        camera.position.set(0, 1.5, 4);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(container.clientWidth, 600);
        renderer.shadowMap.enabled = true;
        container.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 1, 0);
        controls.update();

        // ライティング強化
        scene.add(new THREE.AmbientLight(0xffffff, 0.4));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(5, 10, 5);
        scene.add(dirLight);

        // マテリアル設定
        const boneMat = new THREE.MeshStandardMaterial({{ color: 0x3399ff, roughness: 0.4, metalness: 0.2 }});
        const jointMat = new THREE.MeshStandardMaterial({{ color: 0xeeeeee, roughness: 0.3 }});

        const meshes = {{}};

        // カプセル形状で骨を作成
        function createLimb(name, thickness) {{
            // CapsuleGeometry(半径, 長さ, キャップ分割, 筒分割)
            // 長さ1で作成し、後で距離(dist)に応じてスケール
            const geometry = new THREE.CapsuleGeometry(thickness, 1, 4, 12);
            geometry.rotateX(-Math.PI / 2);
            geometry.translate(0, 0, 0.5); // 始点を原点に
            
            const mesh = new THREE.Mesh(geometry, boneMat);
            scene.add(mesh);
            meshes[name] = mesh;
        }}
        
        function createJoint(index, radius) {{
            let r = radius;
            // 主要な関節（腰・膝）を少し大きく
            if ([23, 24, 25, 26].includes(index)) r *= 1.1;
            const geo = new THREE.SphereGeometry(r, 20, 20);
            const mesh = new THREE.Mesh(geo, jointMat);
            scene.add(mesh);
            meshes['joint_' + index] = mesh;
        }}

        // 接続定義: [始点, 終点, 名前, 太さ]
        const connections = [
            [11, 12, 'shoulders', 0.05],
            [11, 23, 'leftSide', 0.06], [12, 24, 'rightSide', 0.06], 
            [23, 24, 'hips', 0.07],
            [11, 13, 'L_Arm', 0.04], [13, 15, 'L_ForeArm', 0.03],
            [12, 14, 'R_Arm', 0.04], [14, 16, 'R_ForeArm', 0.03],
            [23, 25, 'L_Thigh', 0.07], [25, 27, 'L_Shin', 0.05],
            [24, 26, 'R_Thigh', 0.07], [26, 28, 'R_Shin', 0.05]
        ];

        connections.forEach(c => createLimb(c[2], c[3]));
        [11,12,13,14,15,16,23,24,25,26,27,28,0].forEach(i => createJoint(i, 0.055)); 
        
        const headMesh = new THREE.Mesh(new THREE.SphereGeometry(0.14, 24, 24), boneMat);
        scene.add(headMesh);
        meshes['head'] = headMesh;

        function updateAvatar() {{
            if (!animData.frames.length) return;
            let fIdx = Math.floor(video.currentTime * animData.fps);
            if (fIdx >= animData.frames.length) fIdx = animData.frames.length - 1;
            
            const rawPts = animData.frames[fIdx];
            if (!rawPts) return;

            // スケーリングと高さ調整
            const pts = rawPts.map(p => new THREE.Vector3(p[0]*2.5, p[1]*2.5 + 1.2, p[2]*2.5));

            // 関節
            for (let i=0; i<33; i++) {{
                const mesh = meshes['joint_' + i];
                if (mesh && pts[i]) mesh.position.copy(pts[i]);
            }}
            if (meshes['head'] && pts[0]) meshes['head'].position.copy(pts[0]);

            // 骨の伸縮と向き
            connections.forEach(c => {{
                const mesh = meshes[c[2]];
                const pA = pts[c[0]];
                const pB = pts[c[1]];
                
                if (mesh && pA && pB) {{
                    mesh.position.copy(pA);
                    mesh.lookAt(pB);
                    const dist = pA.distanceTo(pB);
                    // カプセルの全高は (本体長さ + 半径*2) なので、
                    // スケール計算で半径分を考慮して調整
                    const radius = c[3];
                    const scaleFactor = Math.max(0.01, dist - radius * 2);
                    mesh.scale.set(1, 1, scaleFactor);
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
            camera.aspect = container.clientWidth / 600;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, 600);
        }});
    </script>
    """
    st.components.v1.html(html_code, height=650)
