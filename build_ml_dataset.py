import numpy as np
import h5py

def build_ml_dataset(h5_path, group='train'):
    """
    适用于 /train/sample_x/data, /train/sample_x/label 结构的 h5 文件。
    返回 X: [样本数, 帧数, 关节数*2], Y: [样本数]
    """
    X_list = []
    Y_list = []
    with h5py.File(h5_path, 'r') as f:
        group_root = f[group]
        for sample_name in group_root:
            sample = group_root[sample_name]
            data = sample['data'][:]    # (frames, joints, 2)
            label = sample['label'][()] # 标量
            feat_seq = data.reshape(data.shape[0], -1) # flatten joints*2
            X_list.append(feat_seq)
            Y_list.append(label)
    X = np.array(X_list)
    # 在 build_ml_dataset 中调用
    X = augment_data(X)
    Y = np.array(Y_list)

    return X, Y


def augment_data(data):
    """对数据进行简单增强，例如添加噪声"""
    noise = np.random.normal(0, 0.01, data.shape)
    return data + noise

