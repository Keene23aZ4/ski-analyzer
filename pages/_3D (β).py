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
        pose_tracker = Pose(
            static_image_mode=False,
            model_complexity=1,   # ★ heavy を使わない
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )


        
        frames_data = []
        prev_pts = None  # ← 正しい位置
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
        
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_small = cv2.resize(rgb, (256, 256))
            
            results = pose_tracker.process(rgb_small)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                frame_pts = []
            
                for i in range(33):
                    p = lm[i]
                    if p.visibility < 0.5:
                        frame_pts.append(None)
                    else:
                        frame_pts.append([p.x, p.y, p.z])

            
                # 欠損補完
                if prev_pts is not None:
                    for i in range(33):
                        if frame_pts[i] is None:
                            frame_pts[i] = prev_pts[i]
            
                prev_pts = frame_pts
            
                # ★ Kalidokit が期待する順番に並べ替え
                ORDER = [
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32
                ]
                ordered_pts = [frame_pts[i] for i in ORDER]
            
                frames_data.append(ordered_pts)

            else:
                if prev_pts is not None:
                    frames_data.append(prev_pts)
                else:
                    frames_data.append([[0,0,0]] * 33)

    cap.release()
    pose_tracker.close()
    
    video_bytes = open(video_path, "rb").read()
    video_b64 = base64.b64encode(video_bytes).decode()
    import requests
    
    vrm_url = "https://raw.githubusercontent.com/Keene23aZ4/ski-analyzer/main/pages/model.vrm"
    vrm_bytes = requests.get(vrm_url).content
    vrm_b64 = base64.b64encode(vrm_bytes).decode()
    payload = json.dumps({"fps": fps, "frames": frames_data})
    
    # --- (省略) ---
    vrm_b64 = base64.b64encode(vrm_bytes).decode()
    payload = json.dumps({"fps": fps, "frames": frames_data})
    
    # f""" ではなく通常の文字列 """ として定義（波括弧をそのまま書ける）
    html_template = """
    <div style="display: flex; flex-direction: column; align-items: center; gap: 15px;">
        <video id="sync_video" width="100%" controls playsinline style="border-radius: 12px; border: 1px solid #ccc;">
            <source src="data:video/mp4;base64,VAR_VIDEO_B64" type="video/mp4">
        </video>
        <div id="container" style="width:100%; height:600px; background:#1c2833; border-radius:12px; overflow:hidden; border: 1px solid #eaeaea;"></div>
    </div>
    
    <!-- ライブラリ読み込み -->
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/kalidokit@1.1.0/dist/kalidokit.umd.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@1.0.11/lib/three-vrm.js"></script>

    <script>
        const video = document.getElementById('sync_video');
        const container = document.getElementById('container');
        const animData = VAR_PAYLOAD;
    
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1c2833);
    
        const camera = new THREE.PerspectiveCamera(40, container.clientWidth/600, 0.1, 100);
        camera.position.set(0, 1.5, 5);
    
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(container.clientWidth, 600);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);
    
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 1, 0);
        controls.enableDamping = true;
    
        scene.add(new THREE.GridHelper(10, 20, 0x0088ff, 0x444444));
        scene.add(new THREE.AmbientLight(0xffffff, 0.7));
        const light = new THREE.DirectionalLight(0xffffff, 1.0);
        light.position.set(1, 5, 3);
        scene.add(light);

        let currentVRM = null;
        
        // --- VRM読み込み処理 ---
        const loader = new THREE.GLTFLoader();
        function base64ToArrayBuffer(base64) {
            const binary = atob(base64);
            const len = binary.length;
            const buffer = new ArrayBuffer(len);
            const view = new Uint8Array(buffer);
            for (let i = 0; i < len; i++) {
                view[i] = binary.charCodeAt(i);
            }
            return buffer;
        }

        try {
            const vrmBuffer = base64ToArrayBuffer("VAR_VRM_B64");
            loader.parse(vrmBuffer, "", (gltf) => {
                // Namespaceの解決 (threevrm か THREE.VRM かを自動判定)
                const VRM_LIB = (typeof threevrm !== 'undefined') ? threevrm : (THREE.VRM ? THREE : null);
                
                if (VRM_LIB && (VRM_LIB.VRM || VRM_LIB.from)) {
                    const vrmFactory = VRM_LIB.VRM || VRM_LIB;
                    vrmFactory.from(gltf).then((vrm) => {
                        currentVRM = vrm;
                        scene.add(vrm.scene);
                        vrm.scene.rotation.y = Math.PI;
                        console.log("VRM Loaded successfully");
                    }).catch(err => console.error("VRM Factory Error:", err));
                } else {
                    console.error("VRM Library not found in expected namespaces");
                }
            }, (err) => console.error("GLTF Parse Error:", err));
        } catch (e) {
            console.error("Base64 Decode Error:", e);
        }

        // --- アニメーション更新処理 ---
        function updateAvatar() {
            if (!currentVRM || !animData || !animData.frames || animData.frames.length === 0) return;
        
            let fIdx = Math.floor(video.currentTime * animData.fps);
            if (fIdx >= animData.frames.length) fIdx = animData.frames.length - 1;
        
            const raw = animData.frames[fIdx];
            if (!raw || !Array.isArray(raw)) return;

            // 重要: null/undefined対策を徹底
            const mpLandmarks = raw.map((p) => {
                if (!p || !Array.isArray(p) || p.length < 3) {
                    return { x: 0, y: 0, z: 0, visibility: 0 };
                }
                return {
                    x: p[0] || 0,
                    y: 1 - (p[1] || 0),
                    z: -(p[2] || 0),
                    visibility: 1
                };
            });

            try {
                // Kalidokitによる計算
                const kalidoPose = Kalidokit.Pose.solve(mpLandmarks, {
                    runtime: "mediapipe",
                });
            
                if (kalidoPose && currentVRM) {
                    Kalidokit.VRMUtils.animateVRM(currentVRM, kalidoPose);
                }
            } catch (err) {
                // 特定のフレームでのエラーを無視して続行
                if (fIdx % 100 === 0) console.warn("Animation skip at frame", fIdx, err);
            }
        }

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            updateAvatar();
            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', () => {
            camera.aspect = container.clientWidth / 600;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, 600);
        });
    </script>
    """
    
    # 最後に .replace() で変数を流し込む
    html_code = html_template.replace("VAR_VIDEO_B64", video_b64)\
                             .replace("VAR_VRM_B64", vrm_b64)\
                             .replace("VAR_PAYLOAD", payload)
    
    st.components.v1.html(html_code, height=1250)
    
    st.components.v1.html(html_code, height=1250)
