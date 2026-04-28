import os
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. 数据标准化器 (StandardScaler)
# ==========================================
class StandardScaler:
    """
    为了防止数据泄露（未来信息泄漏），仅使用训练数据进行 fit，
    并用其均值与标准差对全量数据进行标准化/反标准化的类
    """
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data_list):
        # 将多个时序数组（仅训练部分）纵向拼接，计算整体均值与方差
        stacked_data = np.vstack(data_list)
        self.mean = stacked_data.mean(axis=0)
        self.std = stacked_data.std(axis=0) + 1e-5

    def transform(self, data_list):
        # 对所有数据进行标准化
        return [(d - self.mean) / self.std for d in data_list]

    def inverse_transform(self, data_tensor):
        """
        反标准化函数：当 Attacker/Observer 希望在原始尺度
        （例如真实温度/交通量）上计算 MSE 时使用
        """
        device = data_tensor.device
        dtype = data_tensor.dtype
        mean_t = torch.tensor(self.mean, device=device, dtype=dtype)
        std_t = torch.tensor(self.std, device=device, dtype=dtype)
        return (data_tensor * std_t) + mean_t

# ==========================================
# 2. PyTorch Dataset（滑动窗口工厂）
# ==========================================
class TSDataset(Dataset):
    def __init__(self, data_list, seq_len, pred_len, flag='test'):
        """
        Args:
            data_list: 形如 [array(L, F), array(L, F), ...] 的时序列表
            flag: 'train' / 'val' / 'test' 之一
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data_list = data_list
        self.valid_indices = [] # 存储 [(文件编号, 起始位置), ...]

        type_map = {'train': 0, 'val': 1, 'test': 2}
        set_type = type_map[flag]

        # 对每个文件划分 Train/Val/Test 的边界
        for i, data in enumerate(data_list):
            length = len(data)
            num_train = int(length * 0.6)
            num_test = int(length * 0.2)
            num_vali = length - num_train - num_test

            # 【重要】时序任务特有的边界设定！
            # 为了做 Test 段的第一个预测，需要回看 seq_len 长度的历史（来自 Val 末尾）
            border1s = [0, num_train - seq_len, length - num_test - seq_len]
            border2s = [num_train, num_train + num_vali, length]

            border1 = border1s[set_type]
            border2 = border2s[set_type]

            # 文件长度过短则跳过
            if border2 - border1 < seq_len + pred_len:
                continue

            # 计算滑动窗口可用的全部索引
            valid_len = (border2 - border1) - seq_len - pred_len + 1
            for j in range(valid_len):
                self.valid_indices.append((i, border1 + j))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 1) 获取应从哪个文件、哪个位置切片
        file_idx, start_pos = self.valid_indices[idx]
        data = self.data_list[file_idx]

        # 2) 计算切片边界
        s_begin = start_pos
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len

        # 3) 切出历史(x) 与未来(y)
        seq_x = data[s_begin:s_end]
        seq_y = data[s_end:r_end]

        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)

# ==========================================
# 3. 获取 DataLoader 的统一 API（main.py 只需要调用它）
# ==========================================
def get_dataloader_and_scaler(data_path, seq_len=96, pred_len=48, batch_size=32, flag='test'):
    """
    读取文件/文件夹数据并标准化，返回指定划分（train/val/test）的 DataLoader
    """
    raw_data_list = []

    # --- 1. 读取数据 ---
    if os.path.isfile(data_path):
        # 单文件（例如 ETTh1.csv）
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            # 只抽取数值列（自动排除日期字符串等）
            data = df.select_dtypes(include=[np.number]).values
            raw_data_list.append(data)
    elif os.path.isdir(data_path):
        # 文件夹（例如 UTSD-full-npy）
        # 搜索文件夹内全部 npy，并按数字顺序（0.npy, 1.npy...）排序后读取
        files = glob.glob(os.path.join(data_path, "**/*.npy"), recursive=True)
        
        def sort_key(f):
            name = os.path.basename(f).split('.')[0]
            return int(name) if name.isdigit() else name
            
        files = sorted(files, key=sort_key)
        for fp in files:
            data = np.load(fp)
            if data.ndim == 1:
                data = data.reshape(-1, 1) # 1D 则转换为 2D (L, 1)
            raw_data_list.append(data)
    
    if not raw_data_list:
        raise ValueError(f"未找到数据: {data_path}")

    # --- 2. 仅用训练数据拟合标准化器（防止数据泄露）---
    scaler = StandardScaler()
    train_chunks = []
    for data in raw_data_list:
        train_len = int(len(data) * 0.6)
        train_chunks.append(data[:train_len])
        
    scaler.fit(train_chunks) # 记录训练段的均值与方差
    scaled_data_list = scaler.transform(raw_data_list) # 应用于全量数据

    # --- 3. 构建 DataLoader ---
    dataset = TSDataset(scaled_data_list, seq_len, pred_len, flag=flag)
    
    # 学术界常用做法：train 打乱；val/test 不打乱
    shuffle = (flag == 'train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    return dataloader, scaler
