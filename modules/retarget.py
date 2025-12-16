import numpy as np
from pygltflib import GLTF2
from pathlib import Path


# ---------------------------------------------------------
# 1. GLB を読み込み、ノード階層とローカル行列を取得
# ---------------------------------------------------------
def load_gltf_nodes(glb_path):
    gltf = GLTF2().load(glb_path)
    nodes = gltf.nodes
    skins = gltf.skins

    # Mixamo モデルは通常 skin が 1 つ
    joints = skins[0].joints

    return nodes, joints


# ---------------------------------------------------------
# 2. ノードのローカル行列を 4x4 numpy に変換
# ---------------------------------------------------------
def node_matrix(node):
    if node.matrix:
        return np.array(node.matrix).reshape(4, 4)

    # TRS から行列を作る
    T = np.eye(4)
    if node.translation:
        T[:3, 3] = node.translation

    R = np.eye(4)
    if node.rotation:
        x, y, z, w = node.rotation
        q = np.array([w, x, y, z])
        # quaternion → rotation matrix
        xx, yy, zz = q[1]**2, q[2]**2, q[3]**2
        xy, xz, yz = q[1]*q[2], q[1]*q[3], q[2]*q[3]
        wx, wy, wz = q[0]*q[1], q[0]*q[2], q[0]*q[3]

        R[:3, :3] = np.array([
            [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
            [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
            [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)]
        ])

    S = np.eye(4)
    if node.scale:
        S[0, 0], S[1, 1], S[2, 2] = node.scale

    return T @ R @ S


# ---------------------------------------------------------
# 3. 親から子へ向かう Tポーズ方向ベクトルを抽出
# ---------------------------------------------------------
def extract_default_dirs(glb_path):
    nodes, joints = load_gltf_nodes(glb_path)

    # ノード index → 子 index の辞書
    children_map = {i: node.children or [] for i, node in enumerate(nodes)}

    # default_dir を入れる辞書
    default_dirs = {}

    # Mixamo の主要ボーン名
    important_bones = [
        "mixamorig:Hips",
        "mixamorig:Spine",
        "mixamorig:Spine1",
        "mixamorig:Spine2",
        "mixamorig:Neck",
        "mixamorig:Head",

        "mixamorig:LeftShoulder",
        "mixamorig:LeftArm",
        "mixamorig:LeftForeArm",
        "mixamorig:LeftHand",

        "mixamorig:RightShoulder",
        "mixamorig:RightArm",
        "mixamorig:RightForeArm",
        "mixamorig:RightHand",

        "mixamorig:LeftUpLeg",
        "mixamorig:LeftLeg",
        "mixamorig:LeftFoot",
        "mixamorig:LeftToeBase",

        "mixamorig:RightUpLeg",
        "mixamorig:RightLeg",
        "mixamorig:RightFoot",
        "mixamorig:RightToeBase",
    ]

    # 名前 → index の辞書
    name_to_index = {node.name: i for i, node in enumerate(nodes)}

    for bone_name in important_bones:
        if bone_name not in name_to_index:
            continue

        parent_idx = name_to_index[bone_name]
        children = children_map[parent_idx]

        if not children:
            continue  # 指先などはスキップ

        child_idx = children[0]  # Mixamo は 1 本の子が基本

        # 親と子のローカル行列
        parent_mat = node_matrix(nodes[parent_idx])
        child_mat = node_matrix(nodes[child_idx])

        # Tポーズの方向ベクトル（ローカル空間）
        parent_pos = parent_mat[:3, 3]
        child_pos = child_mat[:3, 3]

        direction = child_pos - parent_pos
        direction = direction / np.linalg.norm(direction)

        default_dirs[bone_name] = direction.tolist()

    return default_dirs


# ---------------------------------------------------------
# 4. 実行例
# ---------------------------------------------------------
if __name__ == "__main__":
    glb_path = "static/avatar.glb"
    dirs = extract_default_dirs(glb_path)

    print("=== DEFAULT DIRS ===")
    for k, v in dirs.items():
        print(k, v)