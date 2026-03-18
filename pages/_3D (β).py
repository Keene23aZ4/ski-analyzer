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

    # ★ ここは f-string 外なので {} のままでOK
    payload = json.dumps({"fps": fps, "frames": frames_data})

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
        scene.background = new THREE.Color(0x1c2833);

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
        gridXY.rotation.x = Math.PI / 2;
        gridXY.position.set(0, 5, -5);
        scene.add(gridXY);

        const gridYZ = new THREE.GridHelper(10, 20, 0x888888, 0xeeeeee);
        gridYZ.rotation.z = Math.PI / 2;
        gridYZ.position.set(-5, 5, 0);
        scene.add(gridYZ);

        scene.add(new THREE.AxesHelper(5));

        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const light = new THREE.DirectionalLight(0xffffff, 0.7);
        light.position.set(5, 10, 5);
        light.castShadow = true;
        scene.add(light);

        const plane = new THREE.Mesh(
            new THREE.PlaneGeometry(20, 20),
            new THREE.ShadowMaterial({{ opacity: 0.1 }})
        );
        plane.rotation.x = -Math.PI / 2;
        plane.receiveShadow = true;
        scene.add(plane);

        const skinMat = new THREE.MeshStandardMaterial({{ color: 0x828282, roughness: 0.4 }});
        const jointMat = new THREE.MeshStandardMaterial({{
            color: 0x00d2ff,
            emissive: 0x00d2ff,
            emissiveIntensity: 0.2
        }});
        const meshes = {{}};
        // ===== パーツ生成（YBot-lite 装甲スーツ系） =====

        function createLimb(name, rStart, rEnd) {{
            const taper = 0.45;  // ★ 装甲スーツ系：太め → 先細り
            const geo = new THREE.CylinderGeometry(rEnd * taper, rStart, 1, 20);
            geo.rotateX(-Math.PI / 2);
            geo.translate(0, 0, 0.5);

            const mesh = new THREE.Mesh(geo, skinMat);
            mesh.castShadow = true;
            scene.add(mesh);
            meshes[name] = mesh;
        }}

        // ===== 関節サイズ（YBot-lite：+10%） =====
        const jointSize = {{
            11: 0.11, 12: 0.11,   // 肩（+10%）
            13: 0.088, 14: 0.088, // 肘
            15: 0.066, 16: 0.066, // 手首
            23: 0.132, 24: 0.132, // 股関節（+10%）
            25: 0.099, 26: 0.099, // 膝
            27: 0.077, 28: 0.077, // 足首
            0:  0.121              // 頭の付け根（+10%）
        }};

        function createJoint(i) {{
            const r = jointSize[i] || 0.05;
            const mesh = new THREE.Mesh(
                new THREE.SphereGeometry(r, 28, 28),
                jointMat
            );
            mesh.castShadow = true;
            scene.add(mesh);
            meshes["j" + i] = mesh;
        }}

        // ===== 四肢の接続定義（太さは装甲スーツ系に合わせて調整） =====
        const conns = [
            [11, 13, "L_upArm", 0.05, 0.075],
            [13, 15, "L_lowArm", 0.04, 0.055],

            [12, 14, "R_upArm", 0.05, 0.075],
            [14, 16, "R_lowArm", 0.04, 0.055],

            [23, 25, "L_thigh", 0.10, 0.13],
            [25, 27, "L_shin",  0.05, 0.09],

            [24, 26, "R_thigh", 0.10, 0.13],
            [26, 28, "R_shin",  0.05, 0.09]
        ];

        // 四肢パーツ生成
        conns.forEach(c => createLimb(c[2], c[3], c[4]));

        // ===== 胴体 3 分割（装甲スーツ系：胸郭大・腹部細・骨盤張る） =====
        createLimb("upperTorso", 0.08, 0.14);  // 肩 → 胸（大きい）
        createLimb("midTorso",   0.05, 0.07);  // 胸 → みぞおち（細い）
        createLimb("lowerTorso", 0.07, 0.11);  // みぞおち → 腰（張る）

        // 関節生成
        [11,12,13,14,15,16,23,24,25,26,27,28,0].forEach(i => createJoint(i));

        // ===== 頭（ヘルメット感：大きめ） =====
        meshes["head"] = new THREE.Mesh(
            new THREE.SphereGeometry(0.15 * 1.75, 32, 32),  // ★ 1.5倍
            skinMat
        );
        scene.add(meshes["head"]);
        function updateAvatar() {{
            if (!animData.frames.length) return;

            let fIdx = Math.floor(video.currentTime * animData.fps);
            if (fIdx >= animData.frames.length) fIdx = animData.frames.length - 1;

            const raw = animData.frames[fIdx];
            if (!raw) return;

            const pts = raw.map(p => new THREE.Vector3(p[0] * 4, p[1] * 4 + 2.5, p[2] * 4));

            // --- joints ---
            for (let i = 0; i < 33; i++) {{
                if (meshes["j" + i]) meshes["j" + i].position.copy(pts[i]);
            }}
            if (meshes["head"]) meshes["head"].position.copy(pts[0]);

            // ===== 胴体3分割（YBot-lite 装甲スーツ系） =====

            const shMid = new THREE.Vector3().addVectors(pts[11], pts[12]).multiplyScalar(0.5);
            const hiMid = new THREE.Vector3().addVectors(pts[23], pts[24]).multiplyScalar(0.5);

            // ★ 胸郭を上寄りに、腹部を長めに
            const chestMid = shMid.clone().lerp(hiMid, 0.18);
            const stomachMid = shMid.clone().lerp(hiMid, 0.55);

            // ★ S字カーブ（装甲スーツの反り）
            chestMid.z += 0.10;
            stomachMid.z -= 0.10;

            // ★ 肩幅・腰幅（装甲スーツの逆三角形）
            const shoulderWidth = pts[11].distanceTo(pts[12]) * 1.25;
            const hipWidth      = pts[23].distanceTo(pts[24]) * 0.90;

            // ★ 胴体の太さ（胸郭大・腹部細・骨盤張る）
            const radUpper = shoulderWidth * 0.32;  // 胸郭：大きい
            const radMid   = shoulderWidth * 0.18;  // 腹部：細い
            const radLower = hipWidth      * 0.28;  // 骨盤：張る

            // ===== ひねり（肩と腰の向き差） =====
            const shoulderVec = new THREE.Vector3().subVectors(pts[12], pts[11]).normalize();
            const hipVec      = new THREE.Vector3().subVectors(pts[24], pts[23]).normalize();

            const twistAngle = shoulderVec.angleTo(hipVec);
            const twistAxis = new THREE.Vector3().crossVectors(shoulderVec, hipVec).normalize();

            // ===== upperTorso（胸郭） =====
            const upper = meshes["upperTorso"];
            if (upper) {{
                upper.position.copy(shMid);
                upper.lookAt(chestMid);
                const dist = shMid.distanceTo(chestMid);

                // ★ 装甲スーツの胸郭：横幅広く・厚みあり
                upper.scale.set(radUpper / 0.08 * 1.2, radUpper / 0.08 * 1.3, dist);
                upper.rotateOnAxis(twistAxis, twistAngle * 0.15);
            }}

            // ===== midTorso（腹部） =====
            const mid = meshes["midTorso"];
            if (mid) {{
                mid.position.copy(chestMid);
                mid.lookAt(stomachMid);
                const dist = chestMid.distanceTo(stomachMid);

                // ★ 腹部は細く・長め
                mid.scale.set(radMid / 0.08 * 0.9, radMid / 0.08 * 0.7, dist);
                mid.rotateOnAxis(twistAxis, twistAngle * 0.45);
            }}

            // ===== lowerTorso（骨盤） =====
            const lower = meshes["lowerTorso"];
            if (lower) {{
                lower.position.copy(stomachMid);
                lower.lookAt(hiMid);
                const dist = stomachMid.distanceTo(hiMid);

                // ★ 骨盤は張る（装甲スーツの腰）
                lower.scale.set(radLower / 0.08 * 1.1, radLower / 0.08 * 1.2, dist);
                lower.rotateOnAxis(twistAxis, twistAngle * 0.75);
            }}
            // ===== 腕の自然形状（YBot-lite 装甲スーツ系） =====
            updateArm("L_upArm",  pts[11], pts[13], shoulderWidth * 0.22,  0.18);
            updateArm("L_lowArm", pts[13], pts[15], shoulderWidth * 0.17, -0.12);

            updateArm("R_upArm",  pts[12], pts[14], shoulderWidth * 0.22, -0.18);
            updateArm("R_lowArm", pts[14], pts[16], shoulderWidth * 0.17,  0.12);

            // ===== 脚の自然形状（YBot-lite 装甲スーツ系） =====
            updateLeg("L_thigh", pts[23], pts[25], hipWidth * 0.25, 0.0);
            updateLeg("L_shin", pts[25], pts[27], hipWidth * 0.20, 0.0);

            updateLeg("R_thigh", pts[24], pts[26], hipWidth * 0.25, 0.0);
            updateLeg("R_shin", pts[26], pts[28], hipWidth * 0.20, 0.0);
        }}

        // ===== 腕の自然形状（装甲スーツ系：太め → 先細り） =====
        function updateArm(name, pA, pB, baseRadius, twist = 0) {{
            const m = meshes[name];
            if (!m) return;

            const length = pA.distanceTo(pB);

            m.position.copy(pA);
            m.lookAt(pB);

            // ★ 装甲スーツの腕：太め → 手首に向かって締まる
            const scaleX = baseRadius * 0.75;
            const scaleY = baseRadius * 0.55;

            m.scale.set(scaleX / 0.05, scaleY / 0.05, length);

            // ★ ひねり（左右差でキャラ性）
            m.rotateZ(twist);
        }}

        // ===== 脚の自然形状（装甲スーツ系：太もも強め） =====
        function updateLeg(name, pA, pB, baseRadius, twist = 0) {{
            const m = meshes[name];
            if (!m) return;

            const length = pA.distanceTo(pB);

            m.position.copy(pA);
            m.lookAt(pB);

            // ★ 太ももは強く、スネは細く
            const scaleX = baseRadius * 0.90;
            const scaleY = baseRadius * 0.70;

            m.scale.set(scaleX / 0.06, scaleY / 0.06, length);

            // ★ ひねり
            m.rotateZ(twist);
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
