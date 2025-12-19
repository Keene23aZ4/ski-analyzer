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



# --- Page Setup ---
st.set_page_config(page_title="3D Plot Avatar", layout="centered")
st.title("3D motion (β)")


uploaded = st.file_uploader("upload your video", type=["mp4", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    with st.spinner("ANALYSING..."):
        # (MediaPipeの処理は前述のコードと同じため省略、frames_dataとfpsを取得済みと仮定)
        # ※実際の実装ではここに前述のPose処理を入れてください
        pass 

    # --- ダミーデータ作成（動作確認用/実際はMediaPipeの結果を使用） ---
    # payload = json.dumps({"fps": fps, "frames": frames_data})

    html_code = f"""
    <div id="container" style="width:100%; height:600px; background:#ffffff; border:1px solid #ccc; border-radius:8px;"></div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        const container = document.getElementById('container');
        const animData = {payload};
        
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xfcfcfc); // プロット風の白背景
        
        const camera = new THREE.PerspectiveCamera(45, container.clientWidth/600, 0.1, 100);
        camera.position.set(5, 5, 8);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(container.clientWidth, 600);
        container.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        
        // --- プロット空間の作成 ---
        
        // 1. 地面グリッド (XZ平面)
        const gridXZ = new THREE.GridHelper(10, 10, 0x888888, 0xcccccc);
        scene.add(gridXZ);
        
        // 2. 壁面グリッド (XY平面)
        const gridXY = new THREE.GridHelper(10, 10, 0x888888, 0xcccccc);
        gridXY.rotation.x = Math.PI / 2;
        gridXY.position.set(0, 5, -5);
        scene.add(gridXY);

        // 3. 壁面グリッド (YZ平面)
        const gridYZ = new THREE.GridHelper(10, 10, 0x888888, 0xcccccc);
        gridYZ.rotation.z = Math.PI / 2;
        gridYZ.position.set(-5, 5, 0);
        scene.add(gridYZ);

        // 4. 座標軸 (RGB = XYZ)
        const axesHelper = new THREE.AxesHelper(5);
        scene.add(axesHelper);

        // ライティング
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
        scene.add(ambientLight);
        const pointLight = new THREE.PointLight(0xffffff, 0.5);
        pointLight.position.set(10, 10, 10);
        scene.add(pointLight);

        // --- アバターパーツ（前述のカプセル/筋肉モデルをここに配置） ---
        // ※骨格生成ロジック(createLimb等)をここに記述
        
        function animate() {{
            requestAnimationFrame(animate);
            // updateAvatar(); // ここで座標更新
            renderer.render(scene, camera);
        }}
        animate();
    </script>
    """
    st.components.v1.html(html_code, height=620)
