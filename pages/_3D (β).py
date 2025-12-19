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


uploaded = st.file_uploader("upload your video", type=["mp4", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    with st.spinner("座標抽出中..."):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # 解析精度を求める場合は model_complexity=2 に変更（ただし重くなります）
        pose_tracker = Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
        
        frames_data = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_tracker.process(rgb)
            if results.pose_world_landmarks:
                lm = results.pose_world_landmarks.landmark
                # Y軸を反転させ、かつ位置調整を行いやすいように生データをリスト化
                frame_pts = [[p.x, -p.y, -p.z] for p in lm]
                frames_data.append(frame_pts)
            else:
                frames_data.append(None)
        cap.release()
        pose_tracker.close()

    # JSに渡すためのJSONデータ（ここで NameError の原因となった payload を定義）
    payload = json.dumps({"fps": fps, "frames": frames_data})

    # Three.js コード
    html_code = f"""
    <div id="container" style="width:100%; height:600px; background:#ffffff; border:1px solid #ccc; border-radius:8px;"></div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        const container = document.getElementById('container');
        const animData = {payload}; // Pythonの変数をJSに埋め込み
        
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xfcfcfc); // プロット風の白背景
        
        const camera = new THREE.PerspectiveCamera(45, container.clientWidth/600, 0.1, 100);
        camera.position.set(4, 4, 6);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(container.clientWidth, 600);
        container.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 1, 0);
        controls.update();

        // --- プロット空間（グリッド）の作成 ---
        // XZ面（底面）
        const gridXZ = new THREE.GridHelper(10, 10, 0x888888, 0xdddddd);
        scene.add(gridXZ);
        
        // XY面（背面）
        const gridXY = new THREE.GridHelper(10, 10, 0x888888, 0xeeeeee);
        gridXY.rotation.x = Math.PI / 2;
        gridXY.position.set(0, 5, -5);
        scene.add(gridXY);

        // YZ面（左側面）
        const gridYZ = new THREE.GridHelper(10, 10, 0x888888, 0xeeeeee);
        gridYZ.rotation.z = Math.PI / 2;
        gridYZ.position.set(-5, 5, 0);
        scene.add(gridYZ);

        // 座標軸
        const axesHelper = new THREE.AxesHelper(5);
        scene.add(axesHelper);

        // ライティング
        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
        dirLight.position.set(5, 10, 5);
        scene.add(dirLight);

        // --- アバター構成パーツ ---
        const skinMat = new THREE.MeshStandardMaterial({{ color: 0x3399ff, roughness: 0.5 }});
        const jointMat = new THREE.MeshStandardMaterial({{ color: 0x666666 }});
        const meshes = {{}};

        function createLimb(name, thickness) {{
            const geo = new THREE.CapsuleGeometry(thickness, 1, 4, 8);
            geo.rotateX(-Math.PI / 2);
            geo.translate(0, 0, 0.5);
            const mesh = new THREE.Mesh(geo, skinMat);
            scene.add(mesh);
            meshes[name] = mesh;
        }}
        
        function createJoint(i, r) {{
            const mesh = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 16), jointMat);
            scene.add(mesh);
            meshes['j' + i] = mesh;
        }}

        // 定義
        const conns = [
            [11,12,'sh',.05],[11,23,'ls',.06],[12,24,'rs',.06],[23,24,'hp',.07],
            [11,13,'la',.04],[13,15,'lf',.03],[12,14,'ra',.04],[14,16,'rf',.03],
            [23,25,'lt',.07],[25,27,'lsn',.05],[24,26,'rt',.07],[26,28,'rsn',.05]
        ];
        conns.forEach(c => createLimb(c[2], c[3]));
        [11,12,13,14,15,16,23,24,25,26,27,28,0].forEach(i => createJoint(i, 0.05));
        meshes['head'] = new THREE.Mesh(new THREE.SphereGeometry(0.14, 20, 20), skinMat);
        scene.add(meshes['head']);

        let startTime = Date.now();
        function updateAvatar() {{
            if (!animData.frames.length) return;
            // ループ再生のための時間計算
            let time = (Date.now() - startTime) / 1000;
            let fIdx = Math.floor(time * animData.fps) % animData.frames.length;
            
            const raw = animData.frames[fIdx];
            if (!raw) return;

            // プロット空間に合わせたスケーリング（中心を原点付近に）
            const pts = raw.map(p => new THREE.Vector3(p[0]*4, p[1]*4 + 2, p[2]*4));

            for (let i=0; i<33; i++) {{
                if (meshes['j'+i]) meshes['j'+i].position.copy(pts[i]);
            }
            if (meshes['head']) meshes['head'].position.copy(pts[0]);

            conns.forEach(c => {{
                const m = meshes[c[2]], pA = pts[c[0]], pB = pts[c[1]];
                if (m && pA && pB) {{
                    m.position.copy(pA);
                    m.lookAt(pB);
                    m.scale.set(1, 1, pA.distanceTo(pB) - c[3]*2);
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
    st.components.v1.html(html_code, height=620)
