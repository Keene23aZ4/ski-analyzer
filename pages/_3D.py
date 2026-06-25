import streamlit as st
import cv2
import numpy as np
import json
import tempfile
import base64
from pathlib import Path
import requests

st.set_page_config(page_title="3D Plot Avatar", layout="centered")
st.title("3D Motion Analysis")

# 背景（任意）
def set_background():
    img_path = Path("static/1704273575813.jpg")
    if img_path.exists():
        encoded = base64.b64encode(img_path.read_bytes()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
        }}
        </style>
        """, unsafe_allow_html=True)

set_background()

uploaded = st.file_uploader("Upload your video", type=["mp4", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    with st.spinner("Analyzing Pose..."):
        import mediapipe as mp
        mp_pose = mp.solutions.pose

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        pose_tracker = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        frames_data = []
        last_valid_frame = [[0, 0, 0] for _ in range(33)]  # ← バグ修正

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_tracker.process(rgb)  # ← 解像度低下防止

            if results.pose_landmarks:
                current_frame = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
                frames_data.append(current_frame)
                last_valid_frame = current_frame
            else:
                frames_data.append(last_valid_frame)

        cap.release()

    video_b64 = base64.b64encode(open(video_path, "rb").read()).decode()
    vrm_url = "https://raw.githubusercontent.com/Keene23aZ4/ski-analyzer/main/pages/model.vrm"
    vrm_b64 = base64.b64encode(requests.get(vrm_url).content).decode()

    payload = json.dumps({"fps": fps, "frames": frames_data})

    html = f"""
    <div style="width:100%; text-align:center;">
        <video id="v" width="100%" controls>
            <source src="data:video/mp4;base64,{video_b64}">
        </video>
        <div id="c" style="width:100%; height:500px;"></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/kalidokit@1.1.0/dist/kalidokit.umd.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@1.0.11/lib/three-vrm.js"></script>

    <script>
    const animData = {payload};
    const video = document.getElementById("v");
    const container = document.getElementById("c");

    let currentVRM;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, container.clientWidth/500, 0.1, 100);
    camera.position.set(0, 1.4, 3);

    const renderer = new THREE.WebGLRenderer({{antialias:true}});
    renderer.setSize(container.clientWidth, 500);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(1,3,2);
    scene.add(light);

    // --- VRMロード ---
    const loader = new THREE.GLTFLoader();
    const binary = atob("{vrm_b64}");
    const buf = new Uint8Array(binary.length);
    for (let i=0; i<binary.length; i++) buf[i] = binary.charCodeAt(i);

    loader.parse(buf.buffer, "", gltf => {{
        THREE.VRM.from(gltf).then(vrm => {{
            currentVRM = vrm;
            scene.add(vrm.scene);

            vrm.scene.rotation.y = 0;
            vrm.scene.position.set(0, -1, 0);
        }});
    }});

    // --- ボーン適用 ---
    const rigRotation = (name, rot, damp=1, lerp=0.3) => {{
        const bone = currentVRM.humanoid.getBoneNode(name);
        if (!bone || !rot) return;

        const euler = new THREE.Euler(
            rot.x * damp,
            rot.y * damp,
            rot.z * damp
        );

        const quat = new THREE.Quaternion().setFromEuler(euler);
        bone.quaternion.slerp(quat, lerp);
    }};

    function updateAvatar() {{
        if (!currentVRM) return;

        const time = video.currentTime * animData.fps;
        const f0 = Math.floor(time);
        const f1 = Math.min(f0+1, animData.frames.length-1);
        const t = time - f0;

        if (!animData.frames[f0]) return;

        // 補間
        const pts = animData.frames[f0].map((p,i)=>[
            p[0]*(1-t)+animData.frames[f1][i][0]*t,
            p[1]*(1-t)+animData.frames[f1][i][1]*t,
            p[2]*(1-t)+animData.frames[f1][i][2]*t
        ]);

        // ✅ 座標変換（最重要修正）
        const mp = pts.map(p => ({{
            x: (p[0]-0.5)*2,
            y: (0.5-p[1])*2,
            z: -(p[2])*2,
            visibility:1
        }}));

        const pose = Kalidokit.Pose.solve(mp, {{runtime:"mediapipe"}});

        if (pose) {{
            rigRotation("Hips", pose.Hips.rotation, 0.7);
            rigRotation("Spine", pose.Spine, 0.7);

            rigRotation("LeftUpperArm", pose.LeftUpperArm);
            rigRotation("LeftLowerArm", pose.LeftLowerArm);
            rigRotation("RightUpperArm", pose.RightUpperArm);
            rigRotation("RightLowerArm", pose.RightLowerArm);

            rigRotation("LeftUpperLeg", pose.LeftUpperLeg);
            rigRotation("LeftLowerLeg", pose.LeftLowerLeg);
            rigRotation("RightUpperLeg", pose.RightUpperLeg);
            rigRotation("RightLowerLeg", pose.RightLowerLeg);
        }}
    }}

    function animate() {{
        requestAnimationFrame(animate);
        updateAvatar();
        controls.update();
        renderer.render(scene,camera);
    }}

    animate();

    window.addEventListener("resize", ()=>{{
        camera.aspect = container.clientWidth / 500;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, 500);
    }});

    </script>
    """

    st.components.v1.html(html, height=1000)
