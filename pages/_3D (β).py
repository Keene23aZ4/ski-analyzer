import streamlit as st
import cv2
import numpy as np
import json
import tempfile
import mediapipe as mp
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

# 既存のヘルパー関数（環境に合わせて読み込み元を確認してください）
# もし retarget.py が手元にない場合は、default_dirs をハードコードする必要がありますが
# ここではユーザーの環境にある前提で進めます。
try:
    from retarget import extract_default_dirs
except ImportError:
    st.error("retarget.py が見つかりません。extract_default_dirs 関数が必要です。")
    st.stop()

# ==========================================
# 1. 数学ヘルパー関数（クォータニオン演算）
# ==========================================

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return v
    return v / norm

def quat_multiply(q1, q2):
    """
    クォータニオンの積 q1 * q2
    q = [x, y, z, w] の順序で扱います (Three.js互換)
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=float)

def quat_conjugate(q):
    """クォータニオンの共役（逆回転）"""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)

def compute_quaternion_from_vectors(u, v):
    """
    ベクトル u を ベクトル v に合わせるための回転クォータニオンを計算
    (Shortest Arc)
    """
    u = normalize(u)
    v = normalize(v)

    dot = np.dot(u, v)
    
    # 平行な場合 (回転なし)
    if dot > 0.999999:
        return np.array([0, 0, 0, 1], dtype=float)
    
    # 反平行の場合 (180度回転) - 軸を適当に決める必要がある
    if dot < -0.999999:
        axis = np.cross(np.array([1, 0, 0]), u)
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(np.array([0, 1, 0]), u)
        axis = normalize(axis)
        # 180度 = pi radians. sin(pi/2)=1, cos(pi/2)=0
        return np.array([axis[0], axis[1], axis[2], 0], dtype=float)

    # 一般的なケース
    axis = np.cross(u, v)
    
    # クォータニオン構築 (q = [x, y, z, w])
    # w = sqrt((|u|*|v|) + u.v) ... Half-way vector trick
    # ここでは簡易的に angle/axis から計算
    # q = [ sin(theta/2)*axis, cos(theta/2) ]
    
    # 簡略化された実装 (axis と dot + 1 を使う方法)
    # q = (u x v, 1 + u . v) then normalize
    qx, qy, qz = axis
    qw = 1.0 + dot
    
    q = np.array([qx, qy, qz, qw])
    return normalize(q)


# ==========================================
# 2. 設定・定数定義
# ==========================================

# ボーンの親子関係定義（Mixamo標準リグ）
HIERARCHY = {
    "mixamorigHips": None,
    
    # Spine Chain
    "mixamorigSpine": "mixamorigHips",
    "mixamorigSpine1": "mixamorigSpine",
    "mixamorigSpine2": "mixamorigSpine1",
    "mixamorigNeck": "mixamorigSpine2",
    "mixamorigHead": "mixamorigNeck",
    
    # Legs
    "mixamorigLeftUpLeg": "mixamorigHips",
    "mixamorigLeftLeg": "mixamorigLeftUpLeg",
    "mixamorigLeftFoot": "mixamorigLeftLeg",
    "mixamorigLeftToeBase": "mixamorigLeftFoot",
    
    "mixamorigRightUpLeg": "mixamorigHips",
    "mixamorigRightLeg": "mixamorigRightUpLeg",
    "mixamorigRightFoot": "mixamorigRightLeg",
    "mixamorigRightToeBase": "mixamorigRightFoot",
    
    # Arms
    "mixamorigLeftArm": "mixamorigSpine2",
    "mixamorigLeftForeArm": "mixamorigLeftArm",
    "mixamorigLeftHand": "mixamorigLeftForeArm",
    
    "mixamorigRightArm": "mixamorigSpine2",
    "mixamorigRightForeArm": "mixamorigRightArm",
    "mixamorigRightHand": "mixamorigRightForeArm",
}

# 計算順序リスト（親から子へ確実に処理するため）
PROCESS_ORDER = [
    "mixamorigHips",
    "mixamorigSpine", "mixamorigSpine1", "mixamorigSpine2",
    "mixamorigNeck", "mixamorigHead",
    "mixamorigLeftUpLeg", "mixamorigLeftLeg", "mixamorigLeftFoot", "mixamorigLeftToeBase",
    "mixamorigRightUpLeg", "mixamorigRightLeg", "mixamorigRightFoot", "mixamorigRightToeBase",
    "mixamorigLeftArm", "mixamorigLeftForeArm", "mixamorigLeftHand",
    "mixamorigRightArm", "mixamorigRightForeArm", "mixamorigRightHand"
]

# MediaPipeのランドマークIDマッピング
MIXAMO_MAP = {
    "mixamorigHips": 23, # 便宜上
    "mixamorigLeftArm": 11,
    "mixamorigLeftForeArm": 13,
    "mixamorigLeftHand": 15,
    "mixamorigRightArm": 12,
    "mixamorigRightForeArm": 14,
    "mixamorigRightHand": 16,
    "mixamorigLeftUpLeg": 23,
    "mixamorigLeftLeg": 25,
    "mixamorigLeftFoot": 27,
    "mixamorigLeftToeBase": 31,
    "mixamorigRightUpLeg": 24,
    "mixamorigRightLeg": 26,
    "mixamorigRightFoot": 28,
    "mixamorigRightToeBase": 32,
}

# ==========================================
# 3. アプリケーション本体
# ==========================================

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
st.set_page_config(page_title="3D Pose → Avatar Motion (Fixed)", layout="wide")

mp_pose = mp.solutions.pose

def extract_3d_pose_sequence(video_path: str, stride: int = 3):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # model_complexity=2 にすると精度が上がりますが重くなります。1でOKならそのままで。
    pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True)

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_world_landmarks:
            landmarks = result.pose_world_landmarks.landmark
            lm_xyz = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks]
        else:
            lm_xyz = [] # 空の場合は後でスキップ処理

        if lm_xyz:
            frames.append({"landmarks": lm_xyz})
        
        frame_idx += 1

    cap.release()
    pose.close()
    return {"frames": frames, "fps": fps}

# -------------------------------------------------------------------
# メイン処理
# -------------------------------------------------------------------

uploaded = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi", "mkv"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpf:
        tmpf.write(uploaded.read())
        tmp_path = tmpf.name

    with st.spinner("MediaPipe解析中..."):
        seq = extract_3d_pose_sequence(tmp_path, stride=3)

    # GLB から Tポーズの default_dir を取得
    model_path = Path("static/avatar.glb")
    if not model_path.exists():
        st.error("static/avatar.glb が見つかりません。")
        st.stop()

    # コロン付きのキーをコロン無しに正規化して読み込み
    default_dirs_raw = extract_default_dirs(str(model_path))
    default_dirs = {k.replace("mixamorig:", "mixamorig"): v for k, v in default_dirs_raw.items()}

    def mp_to_vec(lm):
        """MediaPipe座標系 -> Three.js/Mixamo座標系への変換"""
        return np.array([lm["x"], -lm["y"], -lm["z"]], dtype=float)

    def convert_to_mixamo_json_hierarchical(frames):
        anim = {"frames": []}

        for f in frames:
            LM = f["landmarks"]
            if not LM: continue
            
            pts = [mp_to_vec(lm) for lm in LM]

            # 主要ポイントの抽出
            hips_pt = pts[23]
            # 左右の腰の中点をHipsとする
            hips_center = (pts[23] + pts[24]) / 2.0
            neck_pt = pts[11] + (pts[12] - pts[11]) / 2.0 # 両肩の中点付近
            nose_pt = pts[0]

            # 簡易スパイン補間
            spine_vec = neck_pt - hips_center
            spine_len = np.linalg.norm(spine_vec)
            
            # 各Spine関節の位置を仮定（直線補間）
            spine_pos = hips_center + spine_vec * 0.25
            spine1_pos = hips_center + spine_vec * 0.50
            spine2_pos = hips_center + spine_vec * 0.75
            
            # --- 1. 各ボーンの「現在の向き（ベクトル）」を定義 ---
            # ここでは「グローバル空間でそのボーンが向いている方向」を辞書にする
            current_vectors = {}
            
            # Spine Chain (上方向)
            current_vectors["mixamorigSpine"]  = spine_pos - hips_center
            current_vectors["mixamorigSpine1"] = spine1_pos - spine_pos
            current_vectors["mixamorigSpine2"] = spine2_pos - spine1_pos
            current_vectors["mixamorigNeck"]   = nose_pt - spine2_pos # Neckは鼻方向へ
            current_vectors["mixamorigHead"]   = np.array([0, 1, 0]) # Headは固定か、鼻の向きなどを利用

            # 手足 (MediaPipeの接続を利用)
            for bone, idx in MIXAMO_MAP.items():
                if bone in current_vectors: continue
                
                # 親idx -> 子idx を決定
                parent_idx = idx
                
                # 特別なケース: 足先
                if bone == "mixamorigLeftToeBase":
                    child_idx = 31 # Left Foot Index -> Toe
                    parent_idx = 27 # Left Ankle
                elif bone == "mixamorigRightToeBase":
                    child_idx = 32
                    parent_idx = 28
                else:
                    # 標準的には +2 (Arm->ForeArm->Hand)
                    child_idx = idx + 2
                    # 配列外参照防止
                    if child_idx >= len(pts): child_idx = idx 
                
                # ベクトル計算
                v = pts[child_idx] - pts[parent_idx]
                current_vectors[bone] = v

            # Hips用: Hipsは回転のルートなので、「Spineの方向」をY軸、「腰の横幅」をX軸のようにみなす
            # 簡易的にSpine方向を向くように設定
            current_vectors["mixamorigHips"] = spine_vec 

            # --- 2. 階層計算 ---
            frame_data = {}
            global_quats = {} # 親のグローバル回転を保存しておく

            # 位置情報 (Hipsのみ) - スケール調整が必要な場合が多い (例: * 100)
            frame_data["mixamorigHips_pos"] = hips_center.tolist()

            for bone in PROCESS_ORDER:
                if bone not in default_dirs:
                    # 定義がないボーンは単位行列
                    frame_data[bone] = [0,0,0,1]
                    global_quats[bone] = np.array([0,0,0,1])
                    continue
                
                # Tポーズ（初期姿勢）でのベクトル
                def_v = np.array(default_dirs[bone], dtype=float)
                
                # 現在のフレームでのベクトル
                # 定義されていない場合は初期姿勢維持
                tgt_v = current_vectors.get(bone, def_v)
                
                # グローバル回転を計算 (Default -> Target)
                q_global = compute_quaternion_from_vectors(def_v, tgt_v)
                
                # 親の回転を取得して「逆回転」を掛ける
                parent_name = HIERARCHY.get(bone)
                
                if parent_name and parent_name in global_quats:
                    q_parent_global = global_quats[parent_name]
                    q_parent_inv = quat_conjugate(q_parent_global)
                    
                    # Local = Parent_Inv * Global
                    q_local = quat_multiply(q_parent_inv, q_global)
                else:
                    # 親がいない(Root)ならそのまま
                    q_local = q_global
                
                # 保存
                global_quats[bone] = q_global # 次の子のためにグローバルを保存
                frame_data[bone] = q_local.tolist() # 出力はローカル

            anim["frames"].append(frame_data)

        return anim

    with st.spinner("モーション変換中..."):
        converted = convert_to_mixamo_json_hierarchical(seq["frames"])
        payload = json.dumps(converted)

    # GLB を base64 で Three.js に渡す
    model_data = base64.b64encode(model_path.read_bytes()).decode()
    video_data = base64.b64encode(open(tmp_path,'rb').read()).decode()

    html_code = """
    <div style="display:flex; flex-direction:column; gap:20px;">
      <div style="flex:1;">
        <h3>オリジナル動画</h3>
        <video id="video" width="100%" controls playsinline>
          <source src="data:video/mp4;base64,VIDEO_B64" type="video/mp4">
        </video>
      </div>
      <div style="flex:1;">
        <h3>3Dアバター再生 (Hierarchical Fix)</h3>
        <div id="container" style="width:100%; height:600px;"></div>
      </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.141.0/examples/js/controls/OrbitControls.js"></script>
    <script>
    let avatar;
    let mixer;

    const container = document.getElementById('container');
    const w = container.clientWidth || window.innerWidth;
    const h = container.clientHeight || 600;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x222222);

    const light = new THREE.HemisphereLight(0xffffff, 0x444444, 1.0);
    light.position.set(0, 20, 0);
    scene.add(light);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight.position.set(5, 10, 7.5);
    scene.add(dirLight);

    const camera = new THREE.PerspectiveCamera(60, w/h, 0.01, 100);
    camera.position.set(0, 1.5, 3);
    
    const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true});
    renderer.setSize(w, h);
    renderer.outputEncoding = THREE.sRGBEncoding;
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1.0, 0);
    controls.update();

    // グリッド
    scene.add(new THREE.GridHelper(10, 10));
    scene.add(new THREE.AxesHelper(1));

    const loader = new THREE.GLTFLoader();
    loader.load("data:application/octet-stream;base64,MODEL_B64", function(gltf){
      avatar = gltf.scene;
      scene.add(avatar);
      
      // モデル内の全ボーン構造を確認（デバッグ用）
      avatar.traverse((node) => {
         if(node.isBone) {
             // console.log(node.name); 
         }
      });
    });

    const video = document.getElementById("video");
    const anim = PAYLOAD_JSON;

    function updatePose() {
      if (!video || video.paused || !avatar) return;
      
      const duration = video.duration;
      if (!duration) return;

      const progress = video.currentTime / duration;
      const totalFrames = anim.frames.length;
      let frameIndex = Math.floor(progress * totalFrames);
      
      if (frameIndex >= totalFrames) frameIndex = totalFrames - 1;
      if (frameIndex < 0) frameIndex = 0;

      const frame = anim.frames[frameIndex];
      if (!frame) return;

      // Hips位置 (スケールが必要ならここで調整)
      const hp = frame["mixamorigHips_pos"];
      // Y座標などがずれる場合はオフセットを加算
      // avatar.position.set(hp[0], hp[1], hp[2]); 
      
      // ボーン回転適用
      for (const boneName in frame) {
          if (!boneName.startsWith("mixamorig") || boneName.endsWith("_pos")) continue;
          
          const bone = avatar.getObjectByName(boneName);
          if (bone) {
              const q = frame[boneName];
              // Three.jsは q=[x,y,z,w]
              bone.quaternion.set(q[0], q[1], q[2], q[3]);
          }
      }
    }

    function animate() {
      requestAnimationFrame(animate);
      updatePose();
      renderer.render(scene, camera);
    }
    animate();
    
    // ウィンドウリサイズ対応
    window.addEventListener('resize', () => {
        const newW = container.clientWidth;
        const newH = container.clientHeight || 600;
        renderer.setSize(newW, newH);
        camera.aspect = newW / newH;
        camera.updateProjectionMatrix();
    });

    </script>
    """

    # プレースホルダー置換
    html_code = html_code.replace("PAYLOAD_JSON", payload)
    html_code = html_code.replace("MODEL_B64", model_data)
    html_code = html_code.replace("VIDEO_B64", video_data)

    st.components.v1.html(html_code, height=750, scrolling=False)
