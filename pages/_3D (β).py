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

st.set_page_config(page_title="Solid Geometric Avatar", layout="centered")
st.title("⛷️ Solid Ski Avatar (Gap Fixed)")
st.caption("関節のズレを完全修正した肉付けモデル")

uploaded = st.file_uploader("スキー動画をアップロード", type=["mp4", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    with st.spinner("座標抽出中..."):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # モデル複雑度1で高速処理
        pose_tracker = Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
        
        frames_data = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_tracker.process(rgb)
            
            if results.pose_world_landmarks:
                # 座標抽出 (Y, Z反転でThree.jsに合わせる)
                lm = results.pose_world_landmarks.landmark
                frame_pts = [[p.x, -p.y, -p.z] for p in lm]
                frames_data.append(frame_pts)
            else:
                frames_data.append(None)
        cap.release()
        pose_tracker.close()

    # データ埋め込み
    video_bytes = open(video_path, 'rb').read()
    video_b64 = base64.b64encode(video_bytes).decode()
    payload = json.dumps({"fps": fps, "frames": frames_data})

    # Three.js コード
    html_code = f"""
    <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
        <video id="sync_video" width="100%" controls playsinline style="border-radius: 8px;">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        <div id="container" style="width:100%; height:1000px; background:#1a1a1a; border-radius:8px;"></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        const video = document.getElementById('sync_video');
        const container = document.getElementById('container');
        const animData = {payload};
        
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a1a); // 暗い背景
        
        const camera = new THREE.PerspectiveCamera(45, container.clientWidth/500, 0.1, 100);
        camera.position.set(0, 1, 3.5);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(container.clientWidth, 500);
        container.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 0.8, 0);
        controls.update();

        // ライティング
        scene.add(new THREE.AmbientLight(0xffffff, 0.5));
        const spotLight = new THREE.SpotLight(0xffffff, 1);
        spotLight.position.set(5, 10, 7);
        scene.add(spotLight);

        // マテリアル
        const boneMat = new THREE.MeshStandardMaterial({{ color: 0x0088ff, roughness: 0.3 }});
        const jointMat = new THREE.MeshStandardMaterial({{ color: 0xffffff, roughness: 0.3 }});

        const meshes = {{}};

        // --- 修正ポイント: ジオメトリの原点を「端っこ」に移動 ---
        function createLimb(name, thickness) {{
            // CylinderGeometry(上半径, 下半径, 高さ) -> 初期状態は縦向き(Y軸)で中心が原点
            const geometry = new THREE.CylinderGeometry(thickness, thickness, 1, 12);
            
            // 1. まず横倒しにする (Z軸向きにする)
            geometry.rotateX(-Math.PI / 2);
            // 2. 原点を「中心」から「始点」にずらす (Z軸方向に0.5移動)
            geometry.translate(0, 0, 0.5);
            
            const mesh = new THREE.Mesh(geometry, boneMat);
            scene.add(mesh);
            meshes[name] = mesh;
        }}
        
        function createJoint(index, radius) {{
            const geo = new THREE.SphereGeometry(radius, 16, 16);
            const mesh = new THREE.Mesh(geo, jointMat);
            scene.add(mesh);
            meshes['joint_' + index] = mesh;
        }}

        // 接続定義
        const connections = [
            [11, 12, 'shoulders', 0.04],
            [11, 23, 'leftSide', 0.06], [12, 24, 'rightSide', 0.06], [23, 24, 'hips', 0.06],
            [11, 13, 'L_Arm', 0.035], [13, 15, 'L_ForeArm', 0.03],
            [12, 14, 'R_Arm', 0.035], [14, 16, 'R_ForeArm', 0.03],
            [23, 25, 'L_Thigh', 0.05], [25, 27, 'L_Shin', 0.04],
            [24, 26, 'R_Thigh', 0.05], [26, 28, 'R_Shin', 0.04]
        ];

        // メッシュ生成
        connections.forEach(c => createLimb(c[2], c[3]));
        // 関節（主要な部分のみ）
        [11,12,13,14,15,16,23,24,25,26,27,28,0].forEach(i => createJoint(i, 0.065)); 
        
        // 頭
        const headMesh = new THREE.Mesh(new THREE.SphereGeometry(0.12, 24, 24), boneMat);
        scene.add(headMesh);
        meshes['head'] = headMesh;

        function updateAvatar() {{
            if (!animData.frames.length) return;
            let fIdx = Math.floor(video.currentTime * animData.fps);
            if (fIdx >= animData.frames.length) fIdx = animData.frames.length - 1;
            
            const rawPts = animData.frames[fIdx];
            if (!rawPts) return;

            // 座標スケーリング (見やすく2倍に)
            const pts = rawPts.map(p => new THREE.Vector3(p[0]*2, p[1]*2 + 1.2, p[2]*2));

            // 1. 関節の更新
            for (let i=0; i<33; i++) {{
                const mesh = meshes['joint_' + i];
                if (mesh && pts[i]) mesh.position.copy(pts[i]);
            }}
            if (meshes['head'] && pts[0]) meshes['head'].position.copy(pts[0]);

            // 2. 骨の更新 (ここが修正版)
            connections.forEach(c => {{
                const idxA = c[0];
                const idxB = c[1];
                const name = c[2];
                const mesh = meshes[name];
                
                if (mesh && pts[idxA] && pts[idxB]) {{
                    const pA = pts[idxA];
                    const pB = pts[idxB];
                    
                    // 始点に配置
                    mesh.position.copy(pA);
                    
                    // 終点の方を向く
                    mesh.lookAt(pB);
                    
                    // 距離に合わせてZ軸(長さ)を伸縮
                    const dist = pA.distanceTo(pB);
                    mesh.scale.set(1, 1, dist);
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
    st.components.v1.html(html_code, height=520)
