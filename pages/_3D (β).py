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

# MediaPipeの読み込み
try:
    import mediapipe as mp
    from mediapipe.python.solutions import pose as mp_pose
    Pose = mp_pose.Pose
except:
    import mediapipe.solutions.pose as mp_pose
    Pose = mp_pose.Pose

st.set_page_config(page_title="Geometric Avatar", layout="centered")
st.title("⛷️ Lightweight 3D Geometric Avatar")
st.caption("サーバー負荷ゼロ・回転計算不要の「肉付け」アプローチ")

uploaded = st.file_uploader("スキー動画をアップロード", type=["mp4", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    # --- 1. MediaPipeによる解析 (座標データの抽出のみ) ---
    with st.spinner("モーション抽出中..."):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # 軽量化のため complexity=1
        pose_tracker = Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
        
        frames_data = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_tracker.process(rgb)
            
            if results.pose_world_landmarks:
                # ランドマーク座標 (33箇所) をそのままリスト化
                # x:右, y:上, z:手前 (Three.js座標系に合わせて符号調整)
                lm = results.pose_world_landmarks.landmark
                frame_pts = []
                for p in lm:
                    frame_pts.append([p.x, -p.y, -p.z]) # YとZを反転
                frames_data.append(frame_pts)
            else:
                frames_data.append(None)
        
        cap.release()
        pose_tracker.close()

    # --- 2. データをJSON化して埋め込み ---
    video_bytes = open(video_path, 'rb').read()
    video_b64 = base64.b64encode(video_bytes).decode()
    payload = json.dumps({"fps": fps, "frames": frames_data})

    # --- 3. Three.jsによるジオメトリックレンダリング ---
    # ここで「点と点を結ぶ円柱」を生成して肉付けします
    html_code = f"""
    <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
        <video id="sync_video" width="100%" controls playsinline style="border-radius: 8px;">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        <div id="container" style="width:100%; height:500px; background:#222; border-radius:8px;"></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        const video = document.getElementById('sync_video');
        const container = document.getElementById('container');
        const animData = {payload};
        
        // シーン設定
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x222222);
        const camera = new THREE.PerspectiveCamera(45, container.clientWidth/500, 0.1, 100);
        camera.position.set(0, 1, 4);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(container.clientWidth, 500);
        container.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 1, 0);
        controls.update();

        // ライト
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(5, 10, 7);
        scene.add(dirLight);

        // --- アバター構成パーツ (Geometric Robot) ---
        const material = new THREE.MeshStandardMaterial({{ color: 0x00aaff, roughness: 0.4 }});
        const jointMat = new THREE.MeshStandardMaterial({{ color: 0xffffff }});
        
        const meshes = {{}}; // 円柱などの管理用

        // 2点間をつなぐ円柱を作成するヘルパー
        function createLimb(name, radius) {{
            // 中心を原点とする円柱ジオメトリ
            const geometry = new THREE.CylinderGeometry(radius, radius, 1, 16);
            geometry.translate(0, 0.5, 0); // ピボットを端に移動
            geometry.rotateX(-Math.PI / 2); // Z軸向きに変更
            const mesh = new THREE.Mesh(geometry, material);
            scene.add(mesh);
            meshes[name] = mesh;
        }}
        
        // 関節の球体を作成
        function createJoint(index, radius) {{
            const geo = new THREE.SphereGeometry(radius, 16, 16);
            const mesh = new THREE.Mesh(geo, jointMat);
            scene.add(mesh);
            meshes['joint_' + index] = mesh;
        }}

        // 定義: MediaPipeのインデックス接続図
        // [開始点, 終了点, パーツ名, 太さ]
        const connections = [
            [11, 12, 'shoulders', 0.03],
            [11, 23, 'leftSide', 0.08],
            [12, 24, 'rightSide', 0.08],
            [23, 24, 'hips', 0.08],
            [11, 13, 'leftUpperArm', 0.04],
            [13, 15, 'leftForeArm', 0.03],
            [12, 14, 'rightUpperArm', 0.04],
            [14, 16, 'rightForeArm', 0.03],
            [23, 25, 'leftThigh', 0.06],
            [25, 27, 'leftShin', 0.05],
            [24, 26, 'rightThigh', 0.06],
            [26, 28, 'rightShin', 0.05]
        ];

        // 初期化
        connections.forEach(c => createLimb(c[2], c[3]));
        [11,12,13,14,15,16,23,24,25,26,27,28,0].forEach(i => createJoint(i, 0.06)); // 関節 + 頭(0)

        // 頭部 (特別扱い)
        const headGeo = new THREE.SphereGeometry(0.12, 32, 32);
        const headMesh = new THREE.Mesh(headGeo, material);
        scene.add(headMesh);
        meshes['head'] = headMesh;

        // --- アニメーション更新処理 ---
        function updateAvatar() {{
            if (!animData.frames.length) return;
            let fIdx = Math.floor(video.currentTime * animData.fps);
            if (fIdx >= animData.frames.length) fIdx = animData.frames.length - 1;
            
            const pts = animData.frames[fIdx];
            if (!pts) return;

            // 1. 関節（球体）の位置更新
            for (let i=0; i<33; i++) {{
                const mesh = meshes['joint_' + i];
                if (mesh) {{
                    // 座標倍率調整 (x2くらいが見やすい)
                    mesh.position.set(pts[i][0]*2, pts[i][1]*2 + 1, pts[i][2]*2);
                }}
            }}
            
            // 頭の位置
            if(meshes['head']) {{
                // 鼻(0)の位置にセット
                meshes['head'].position.set(pts[0][0]*2, pts[0][1]*2 + 1, pts[0][2]*2);
            }}

            // 2. 骨（円柱）の位置と向き更新
            connections.forEach(c => {{
                const idxA = c[0];
                const idxB = c[1];
                const name = c[2];
                const mesh = meshes[name];
                
                if (mesh) {{
                    const pA = new THREE.Vector3(pts[idxA][0]*2, pts[idxA][1]*2 + 1, pts[idxA][2]*2);
                    const pB = new THREE.Vector3(pts[idxB][0]*2, pts[idxB][1]*2 + 1, pts[idxB][2]*2);
                    
                    // 位置: 始点に合わせる
                    mesh.position.copy(pA);
                    
                    // 向き: 始点から終点を見る
                    mesh.lookAt(pB);
                    
                    // 長さ: 距離に合わせてスケール
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
        
        // リサイズ対応
        window.addEventListener('resize', () => {{
            const w = container.clientWidth;
            camera.aspect = w / 500;
            camera.updateProjectionMatrix();
            renderer.setSize(w, 500);
        }});
    </script>
    """
    st.components.v1.html(html_code, height=520)
