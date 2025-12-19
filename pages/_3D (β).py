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
    # (前略: ライブラリのインポートやMediaPipeの解析部分はこれまでのコードを維持)

    # JavaScriptに渡すデータ
    payload = json.dumps({"fps": fps, "frames": frames_data})

    # (前略: MediaPipeの処理とpayloadの定義までは共通)

    # (前略: MediaPipeの処理とpayloadの定義までは共通)

    html_code = f"""
    <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
        <video id="sync_video" width="100%" controls playsinline style="border-radius: 8px; border: 1px solid #ddd;">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        <div id="container" style="width:100%; height:600px; background:#ffffff; border-radius:8px; border: 1px solid #ccc;"></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        const video = document.getElementById('sync_video');
        const container = document.getElementById('container');
        const animData = {payload}; 
        
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xfcfcfc); // プロット風の明るい背景
        
        const camera = new THREE.PerspectiveCamera(45, container.clientWidth/600, 0.1, 100);
        camera.position.set(5, 5, 8);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(container.clientWidth, 600);
        renderer.shadowMap.enabled = true; // 影を有効化
        container.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 1, 0);
        controls.update();

        // --- 3Dプロット空間の構築 ---

        // 1. 地面 (XZ平面) - メングリッドとサブグリッド
        const gridXZ = new THREE.GridHelper(10, 10, 0x444444, 0xdddddd);
        scene.add(gridXZ);
        
        // 2. 背面 (XY平面)
        const gridXY = new THREE.GridHelper(10, 10, 0x888888, 0xeeeeee);
        gridXY.rotation.x = Math.PI / 2;
        gridXY.position.set(0, 5, -5);
        scene.add(gridXY);

        // 3. 側面 (YZ平面)
        const gridYZ = new THREE.GridHelper(10, 10, 0x888888, 0xeeeeee);
        gridYZ.rotation.z = Math.PI / 2;
        gridYZ.position.set(-5, 5, 0);
        scene.add(gridYZ);

        // 4. 座標軸 (X:赤, Y:緑, Z:青)
        const axesHelper = new THREE.AxesHelper(5);
        scene.add(axesHelper);

        // ライティング (影が出るように設定)
        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const light = new THREE.DirectionalLight(0xffffff, 0.7);
        light.position.set(5, 10, 5);
        light.castShadow = true;
        scene.add(light);

        // 影を受ける透明な地面
        const plane = new THREE.Mesh(
            new THREE.PlaneGeometry(20, 20),
            new THREE.ShadowMaterial({{ opacity: 0.1 }})
        );
        plane.rotation.x = -Math.PI / 2;
        plane.receiveShadow = true;
        scene.add(plane);

        // --- アバター設定 (テーパー円柱 & 左右色分け) ---
        const matL = new THREE.MeshStandardMaterial({{ color: 0xff4444, roughness: 0.4 }}); // 左:赤
        const matR = new THREE.MeshStandardMaterial({{ color: 0x4444ff, roughness: 0.4 }}); // 右:青
        const matC = new THREE.MeshStandardMaterial({{ color: 0x666666, roughness: 0.4 }}); // 中央:グレー
        const meshes = {{}};

        function createLimb(name, r1, r2, side) {{
            const geo = new THREE.CylinderGeometry(r2, r1, 1, 16);
            geo.rotateX(-Math.PI / 2); geo.translate(0, 0, 0.5);
            let mat = matC;
            if(side === 'L') mat = matL;
            if(side === 'R') mat = matR;
            const mesh = new THREE.Mesh(geo, mat);
            mesh.castShadow = true;
            scene.add(mesh);
            meshes[name] = mesh;
        }}

        const conns = [
            [11,12,'sh',.05,.05,'C'], [23,24,'hp',.07,.07,'C'],
            [11,13,'la1',.05,.04,'L'], [13,15,'la2',.04,.02,'L'],
            [12,14,'ra1',.05,.04,'R'], [14,16,'ra2',.04,.02,'R'],
            [23,25,'ll1',.08,.06,'L'], [25,27,'ll2',.06,.03,'L'],
            [24,26,'rl1',.08,.06,'R'], [26,28,'rl2',.06,.03,'R']
        ];
        conns.forEach(c => createLimb(c[2], c[3], c[4], c[5]));

        function updateAvatar() {{
            if (!animData.frames.length) return;
            let fIdx = Math.floor(video.currentTime * animData.fps);
            if (fIdx >= animData.frames.length) fIdx = animData.frames.length - 1;
            const raw = animData.frames[fIdx];
            if (!raw) return;

            // スケールをプロット空間(10x10)に合わせて調整
            const pts = raw.map(p => new THREE.Vector3(p[0]*4, p[1]*4 + 2.5, p[2]*4));

            conns.forEach(c => {{
                const m = meshes[c[2]], p1 = pts[c[0]], p2 = pts[c[1]];
                if (m && p1 && p2) {{
                    m.position.copy(p1);
                    m.lookAt(p2);
                    m.scale.set(1, 1, p1.distanceTo(p2));
                }}
            }});
        }}

        function animate() {{
            requestAnimationFrame(animate);
            updateAvatar();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
    """
    st.components.v1.html(html_code, height=1050)
