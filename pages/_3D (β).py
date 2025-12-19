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

    html_code = f"""
    <div style="display: flex; flex-direction: column; align-items: center; gap: 15px;">
        <video id="sync_video" width="100%" controls playsinline style="border-radius: 12px; border: 1px solid #ccc;">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        <div id="container" style="width:100%; height:550px; background:#ffffff; border-radius:12px; overflow:hidden; border: 1px solid #eaeaea;"></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        const video = document.getElementById('sync_video');
        const container = document.getElementById('container');
        const animData = {payload}; 
        
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf7f9fc);
        
        const camera = new THREE.PerspectiveCamera(40, container.clientWidth/550, 0.1, 100);
        camera.position.set(5, 4, 7);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(container.clientWidth, 550);
        renderer.shadowMap.enabled = true;
        container.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        // ライティング
        scene.add(new THREE.AmbientLight(0xffffff, 0.5));
        const spotLight = new THREE.SpotLight(0xffffff, 0.8);
        spotLight.position.set(5, 10, 5);
        spotLight.castShadow = true;
        scene.add(spotLight);

        // 床とグリッド
        const plane = new THREE.Mesh(new THREE.PlaneGeometry(20, 20), new THREE.ShadowMaterial({{ opacity: 0.1 }}));
        plane.rotation.x = -Math.PI / 2;
        plane.receiveShadow = true;
        scene.add(plane);
        scene.add(new THREE.GridHelper(10, 20, 0x0088ff, 0xdddddd));

        const skinMat = new THREE.MeshStandardMaterial({{ color: 0x2c3e50, roughness: 0.4, metalness: 0.2 }});
        const jointMat = new THREE.MeshStandardMaterial({{ color: 0x00d2ff, emissive: 0x00d2ff, emissiveIntensity: 0.3 }});
        const meshes = {{}};

        // --- テーパー（先細り）円柱を作成する関数 ---
        function createLimb(name, radStart, radEnd) {{
            // CylinderGeometry(上半径, 下半径, 高さ, 分割数)
            // MediaPipeの接続順に合わせて、上が「始点側」、下が「終点側」になるよう設定
            const geo = new THREE.CylinderGeometry(radEnd, radStart, 1, 16);
            geo.rotateX(-Math.PI / 2);
            geo.translate(0, 0, 0.5); // 原点を円柱の端に移動
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

        // 接続定義: [始点INDEX, 終点INDEX, 名前, 始点半径, 終点半径]
        const conns = [
            [11, 12, 'shoulder', 0.05, 0.05],   // 肩
            [23, 24, 'hip', 0.07, 0.07],        // 腰
            [11, 23, 'L_torso', 0.05, 0.07],    // 左脇腹
            [12, 24, 'R_torso', 0.05, 0.07],    // 右脇腹
            [11, 13, 'L_upArm', 0.05, 0.035],   // 左上腕（太→細）
            [13, 15, 'L_lowArm', 0.035, 0.02],  // 左前腕（中→細）
            [12, 14, 'R_upArm', 0.05, 0.035],   // 右上腕
            [14, 16, 'R_lowArm', 0.035, 0.02],  // 右前腕
            [23, 25, 'L_thigh', 0.08, 0.06],    // 左太もも（太→中）
            [25, 27, 'L_shin', 0.06, 0.035],    // 左すね（中→細）
            [24, 26, 'R_thigh', 0.08, 0.06],    // 右太もも
            [26, 28, 'R_shin', 0.06, 0.035]     // 右すね
        ];

        conns.forEach(c => createLimb(c[2], c[3], c[4]));
        [11,12,13,14,15,16,23,24,25,26,27,28,0].forEach(i => createJoint(i, 0.05));
        
        meshes['head'] = new THREE.Mesh(new THREE.SphereGeometry(0.15, 32, 32), skinMat);
        scene.add(meshes['head']);

        function updateAvatar() {{
            if (!animData.frames.length) return;
            let fIdx = Math.floor(video.currentTime * animData.fps);
            if (fIdx >= animData.frames.length) fIdx = animData.frames.length - 1;
            const raw = animData.frames[fIdx];
            if (!raw) return;

            const pts = raw.map(p => new THREE.Vector3(p[0]*3.5, p[1]*3.5 + 2.0, p[2]*3.5));

            // 関節の更新
            for (let i=0; i<33; i++) {{
                if (meshes['j'+i]) meshes['j'+i].position.copy(pts[i]);
            }}
            if (meshes['head']) meshes['head'].position.copy(pts[0]);

            // テーパー円柱の更新
            conns.forEach(c => {{
                const m = meshes[c[2]];
                const pA = pts[c[0]]; // 付け根
                const pB = pts[c[1]]; // 先端
                if (m && pA && pB) {{
                    m.position.copy(pA);
                    m.lookAt(pB);
                    // 距離に合わせてスケール
                    const dist = pA.distanceTo(pB);
                    m.scale.set(1, 1, dist);
                }}
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
    st.components.v1.html(html_code, height=1150)
