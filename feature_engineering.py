import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

def calc_joint_angles(frames, joint_triplets):
    """
    计算每帧指定关节三元组的夹角（弧度）
    joint_triplets: [(a,b,c), ...] 表示以b为顶点的a-b-c夹角
    frames: (帧数, 关节数*2)
    """
    n_frames = frames.shape[0]
    n_joints = frames.shape[1] // 2
    coords = frames.reshape(n_frames, n_joints, 2)
    angles = []
    for a, b, c in joint_triplets:
        ba = coords[:, a] - coords[:, b]
        bc = coords[:, c] - coords[:, b]
        dot = np.sum(ba * bc, axis=1)
        norm_ba = np.linalg.norm(ba, axis=1)
        norm_bc = np.linalg.norm(bc, axis=1)
        cos_angle = dot / (norm_ba * norm_bc + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) # 防止数值溢出
        angles.append(angle)
    return np.stack(angles, axis=1) # [帧数, 角度数]

def calc_skeleton_center(frames):
    """
    计算每帧骨架重心（所有关节坐标均值）
    """
    n_joints = frames.shape[1] // 2
    coords = frames.reshape(frames.shape[0], n_joints, 2)
    center = np.mean(coords, axis=1) # [帧数, 2]
    return center

def calc_symmetry(frames, left_joints, right_joints):
    """
    计算每帧左右对称关节的欧氏距离
    """
    n_joints = frames.shape[1] // 2
    coords = frames.reshape(frames.shape[0], n_joints, 2)
    dists = np.linalg.norm(coords[:, left_joints] - coords[:, right_joints], axis=2)
    return dists # [帧数, len(left_joints)]

def calculate_joint_distances(frames, joint_pairs):
    n_frames = frames.shape[0]
    n_joints = frames.shape[1] // 2
    coords = frames.reshape(n_frames, n_joints, 2)
    dist_features = []
    for idx1, idx2 in joint_pairs:
        dist = np.linalg.norm(coords[:, idx1] - coords[:, idx2], axis=1)
        dist_features.append(dist)
    return np.stack(dist_features, axis=1)

def extract_advanced_features(X, n_segments=3):
    """
    对每个样本提取丰富的骨架时序特征并分段统计。
    """
    # COCO 17点骨架关节对（可根据你数据集微调）
    joint_pairs = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    # 关节三元组（如左肩-左肘-左腕等，夹角特征）
    joint_triplets = [
        (5, 7, 9),  # 左肩-左肘-左腕
        (6, 8, 10), # 右肩-右肘-右腕
        (11, 13, 15), # 左髋-左膝-左踝
        (12, 14, 16), # 右髋-右膝-右踝
    ]
    # 对称关节索引（你可根据数据实际调整）
    left_joints = [5,7,9,11,13,15]
    right_joints = [6,8,10,12,14,16]

    seg_feats = []
    for idx, x in enumerate(X):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        n_frames, n_feats = x.shape
        # 1. 基础坐标
        basic = x
        # 2. 速度、加速度
        velocity = np.diff(x, axis=0, prepend=x[[0]])
        acceleration = np.diff(velocity, axis=0, prepend=velocity[[0]])
        # 3. 距离
        distances = calculate_joint_distances(x, joint_pairs)
        dist_velocity = np.diff(distances, axis=0, prepend=distances[[0]])
        # 4. 角度
        angles = calc_joint_angles(x, joint_triplets)
        angle_velocity = np.diff(angles, axis=0, prepend=angles[[0]])
        # 5. 重心轨迹
        center = calc_skeleton_center(x)
        center_velocity = np.diff(center, axis=0, prepend=center[[0]])
        center_acceleration = np.diff(center_velocity, axis=0, prepend=center_velocity[[0]])
        # 6. 左右对称性
        symmetry = calc_symmetry(x, left_joints, right_joints)
        symmetry_velocity = np.diff(symmetry, axis=0, prepend=symmetry[[0]])

        # 合并所有特征
        all_feat = np.concatenate([
            basic,
            velocity,
            acceleration,
            distances,
            dist_velocity,
            angles,
            angle_velocity,
            center,
            center_velocity,
            center_acceleration,
            symmetry,
            symmetry_velocity
        ], axis=1)
        # 分段统计
        if n_frames < n_segments:
            pad_width = ((0, n_segments - n_frames), (0, 0))
            all_feat = np.pad(all_feat, pad_width, mode='edge')
        segs = np.array_split(all_feat, n_segments, axis=0)
        feats = []
        for seg in segs:
            if seg.size == 0:
                feats.extend([np.zeros(all_feat.shape[1])] * 4)
            else:
                feats.extend([
                    np.mean(seg, axis=0),
                    np.std(seg, axis=0),
                    np.max(seg, axis=0),
                    np.min(seg, axis=0)
                ])
        feats_concat = np.concatenate(feats)
        seg_feats.append(feats_concat)
    return np.array(seg_feats)

def select_best_features(X_train, y_train, X_val, k=300):
    selector = SelectKBest(f_classif, k=min(k, X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel = selector.transform(X_val)
    return X_train_sel, X_val_sel, selector

def get_feature_names(X_shape, n_segments=3, n_basic_feats=34, joint_pair_num=10, angle_num=4, symmetry_num=6):
    """
    可选：返回特征名称（辅助特征重要性可视化）
    """
    stat_names = ['mean','std','max','min']
    feat_names = []
    for seg in range(n_segments):
        for stat in stat_names:
            # basic坐标
            feat_names += [f'seg{seg}_{stat}_joint_{i}' for i in range(n_basic_feats)]
            # velocity、acceleration
            feat_names += [f'seg{seg}_{stat}_vel_{i}' for i in range(n_basic_feats)]
            feat_names += [f'seg{seg}_{stat}_acc_{i}' for i in range(n_basic_feats)]
            # distance, dist_velocity
            feat_names += [f'seg{seg}_{stat}_dist_{i}' for i in range(joint_pair_num)]
            feat_names += [f'seg{seg}_{stat}_distvel_{i}' for i in range(joint_pair_num)]
            # angles, angle_velocity
            feat_names += [f'seg{seg}_{stat}_angle_{i}' for i in range(angle_num)]
            feat_names += [f'seg{seg}_{stat}_anglevel_{i}' for i in range(angle_num)]
            # center, center_velocity, center_acceleration
            feat_names += [f'seg{seg}_{stat}_center_{i}' for i in range(2)]
            feat_names += [f'seg{seg}_{stat}_centervel_{i}' for i in range(2)]
            feat_names += [f'seg{seg}_{stat}_centeracc_{i}' for i in range(2)]
            # symmetry, symmetry_velocity
            feat_names += [f'seg{seg}_{stat}_symmetry_{i}' for i in range(symmetry_num)]
            feat_names += [f'seg{seg}_{stat}_symmetryvel_{i}' for i in range(symmetry_num)]
    return feat_names[:X_shape[-1]]