import streamlit as st
import cv2
import json
import tempfile
import base64
from pathlib import Path

# 背景設定（任意）
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

# MediaPipe 読み込み
try:
    import mediapipe as mp
    from mediapipe.python.solutions import pose as mp_pose
    Pose = mp_pose.Pose
except:
    import mediapipe.solutions.pose as mp_pose
    Pose = mp_pose.Pose

# --- Page Setup ---
st.set_page_config(page_title="3D VRM Motion Analysis", layout="centered")
st.title("3D Motion Analysis with VRM Avatar")

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
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
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

    # --- HTML + JS (VRM対応版・中括弧エスケープ済) ---
    html_code = f"""
    <div style="display:flex; flex-direction:column; align-items:center; gap:15px;">
        <video id="sync_video" width="100%" controls playsinline style="border-radius:12px; border:1px solid #ccc;">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        <div id="container" style="width:100%; height:600px; background:#ffffff; border-radius:12px; overflow:hidden; border:1px solid #eaeaea;"></div>
    </div>

    <!-- Three.js & VRM -->
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@2.0.0/lib/three-vrm.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/controls/OrbitControls.js"></script>

    <script>
        const video = document.getElementById('sync_video');
        const container = document.getElementById('container');
        const animData = {payload};

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1c2833);

        const camera = new THREE.PerspectiveCamera(40, container.clientWidth/600, 0.1, 100);
        camera.position.set(6, 4, 8);

        const renderer = new THREE.WebGLRenderer({{ antialias:true, alpha:true }});
        renderer.setSize(container.clientWidth, 600);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        scene.add(new THREE.GridHelper(10, 20, 0x0088ff, 0xdddddd));
        scene.add(new THREE.AxesHelper(5));

        scene.add(new THREE.AmbientLight(0xffffff, 0.7));
        const light = new THREE.DirectionalLight(0xffffff, 0.8);
        light.position.set(5, 10, 5);
        scene.add(light);

        // --- VRMモデル読み込み ---
        let currentVrm = null;
        const loader = new THREE.GLTFLoader();
        loader.crossOrigin = "anonymous";

        loader.load(
            "model.vrm",
            (gltf) => {{
                THREE.VRMUtils.removeUnnecessaryJoints(gltf.scene);
                THREE.VRM.from(gltf).then((vrm) => {{
                    currentVrm = vrm;
                    scene.add(vrm.scene);
                }});
            }}
        );

        // --- ボーン回転適用 ---
        function applyBoneRotation(bone, pA, pB) {{
            if (!bone) return;
            const dir = new THREE.Vector3().subVectors(pB, pA).normalize();
            const quat = new THREE.Quaternion();
            // VRMのデフォルト姿勢（Y軸下向き）を基準に方向ベクトルへ回転
            quat.setFromUnitVectors(new THREE.Vector3(0, -1, 0), dir);
            bone.quaternion.slerp(quat, 0.4);
        }}

        // --- VRMアバター更新 ---
        function updateAvatar() {{
            if (!currentVrm) return;

            let fIdx = Math.floor(video.currentTime * animData.fps);
            if (fIdx >= animData.frames.length) fIdx = animData.frames.length - 1;

            const raw = animData.frames[fIdx];
            if (!raw) return;

            const pts = raw.map(p => new THREE.Vector3(p[0]*4, p[1]*4 + 2.5, p[2]*4));

            const h = currentVrm.humanoid;

            // 腕
            applyBoneRotation(h.getBoneNode("leftUpperArm"),  pts[11], pts[13]);
            applyBoneRotation(h.getBoneNode("leftLowerArm"),  pts[13], pts[15]);
            applyBoneRotation(h.getBoneNode("rightUpperArm"), pts[12], pts[14]);
            applyBoneRotation(h.getBoneNode("rightLowerArm"), pts[14], pts[16]);

            // 脚
            applyBoneRotation(h.getBoneNode("leftUpperLeg"),  pts[23], pts[25]);
            applyBoneRotation(h.getBoneNode("leftLowerLeg"),  pts[25], pts[27]);
            applyBoneRotation(h.getBoneNode("rightUpperLeg"), pts[24], pts[26]);
            applyBoneRotation(h.getBoneNode("rightLowerLeg"), pts[26], pts[28]);

            // 頭（ざっくり肩〜頭方向）
            applyBoneRotation(h.getBoneNode("head"), pts[11], pts[0]);
        }}

        // --- レンダリング ---
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();

            if (currentVrm) {{
                currentVrm.update(1/60);
            }}

            updateAvatar();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
    """

    st.components.v1.html(html_code, height=1250)
