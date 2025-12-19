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

    video_bytes = open(video_path, 'rb').read()
    video_b64 = base64.b64encode(video_bytes).decode()
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
    
    <script>
        const video = document.getElementById('sync_video');
        const container = document.getElementById('container');
        const animData = {payload}; 
        
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xfcfcfc);
        
        const camera = new THREE.PerspectiveCamera(40, container.clientWidth/600, 0.1, 100);
        camera.position.set(6, 4, 8);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(container.clientWidth, 600);
        renderer.shadowMap.enabled = true;
        container.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        scene.add(new THREE.GridHelper(10, 20, 0x0088ff, 0xdddddd));
        const gridXY = new THREE.GridHelper(10, 20, 0x888888, 0xeeeeee);
        gridXY.rotation.x = Math.PI / 2; gridXY.position.set(0, 5, -5); scene.add(gridXY);
        const gridYZ = new THREE.GridHelper(10, 20, 0x888888, 0xeeeeee);
        gridYZ.rotation.z = Math.PI / 2; gridYZ.position.set(-5, 5, 0); scene.add(gridYZ);
        scene.add(new THREE.AxesHelper(5));

        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const light = new THREE.DirectionalLight(0xffffff, 0.7);
        light.position.set(5, 10, 5); light.castShadow = true; scene.add(light);

        const plane = new THREE.Mesh(new THREE.PlaneGeometry(20, 20), new THREE.ShadowMaterial({{ opacity: 0.1 }}));
        plane.rotation.x = -Math.PI / 2; plane.receiveShadow = true; scene.add(plane);

        const skinMat = new THREE.MeshStandardMaterial({{ color: 0x2c3e50, roughness: 0.4 }});
        const jointMat = new THREE.MeshStandardMaterial({{ color: 0x00d2ff, emissive: 0x00d2ff, emissiveIntensity: 0.2 }});
        const meshes = {{}};

        function createLimb(name, rStart, rEnd) {{
            const geo = new THREE.CylinderGeometry(rEnd, rStart, 1, 16);
            geo.rotateX(-Math.PI / 2);
            geo.translate(0, 0, 0.5);
            const mesh = new THREE.Mesh(geo, skinMat);
            mesh.castShadow = true;
            scene.add(mesh);
            meshes[name] = mesh;
        }}
        
        function createJoint(i, r) {{
            const mesh = new THREE.Mesh(new THREE.SphereGeometry(r, 24, 24), jointMat);
            mesh.castShadow = true;
            scene.add(mesh);
            meshes['j' + i] = mesh;
        }}

        const conns = [
            [11, 13, 'L_upArm', 0.05, 0.035], [13, 15, 'L_lowArm', 0.035, 0.02],
            [12, 14, 'R_upArm', 0.05, 0.035], [14, 16, 'R_lowArm', 0.035, 0.02],
            [23, 25, 'L_thigh', 0.08, 0.06],  [25, 27, 'L_shin', 0.06, 0.035],
            [24, 26, 'R_thigh', 0.08, 0.06],  [26, 28, 'R_shin', 0.06, 0.035]
        ];

        conns.forEach(c => createLimb(c[2], c[3], c[4]));
        createLimb('torso', 0.04, 0.08); 
        
        [11,12,13,14,15,16,23,24,25,26,27,28,0].forEach(i => createJoint(i, 0.05));
        meshes['head'] = new THREE.Mesh(new THREE.SphereGeometry(0.15, 32, 32), skinMat);
        scene.add(meshes['head']);

        function updateAvatar() {{
            if (!animData.frames.length) return;
            let fIdx = Math.floor(video.currentTime * animData.fps);
            if (fIdx >= animData.frames.length) fIdx = animData.frames.length - 1;
            const raw = animData.frames[fIdx];
            if (!raw) return;

            const pts = raw.map(p => new THREE.Vector3(p[0]*4, p[1]*4 + 2.5, p[2]*4));

            for (let i=0; i<33; i++) {{ if (meshes['j'+i]) meshes['j'+i].position.copy(pts[i]); }}
            if (meshes['head']) meshes['head'].position.copy(pts[0]);

            const shMid = new THREE.Vector3().addVectors(pts[11], pts[12]).multiplyScalar(0.5);
            const hiMid = new THREE.Vector3().addVectors(pts[23], pts[24]).multiplyScalar(0.5);
            const shoulderWidth = pts[11].distanceTo(pts[12]);
            const dynamicRadTop = shoulderWidth * 0.225; 

            if (meshes['torso']) {{
                meshes['torso'].position.copy(shMid);
                meshes['torso'].lookAt(hiMid);
                const dist = shMid.distanceTo(hiMid);
                meshes['torso'].scale.set(dynamicRadTop / 0.08, dynamicRadTop / 0.08, dist);
            }}

            conns.forEach(c => {{
                const m = meshes[c[2]], pA = pts[c[0]], pB = pts[c[1]];
                if (m) {{ m.position.copy(pA); m.lookAt(pB); m.scale.set(1, 1, pA.distanceTo(pB)); }}
            }});
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
