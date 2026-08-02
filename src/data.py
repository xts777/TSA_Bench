import os
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. Data standardizer
# ==========================================
class StandardScaler:
    """
    A scaler that fits only on training data to avoid leakage from future information,
    then uses the training mean and standard deviation to standardize and inverse-standardize all data.
    """
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data_list):
        stacked_data = np.vstack(data_list)
        self.mean = stacked_data.mean(axis=0)
        self.std = stacked_data.std(axis=0) + 1e-5

    def transform(self, data_list):
        # Standardize all data.
        return [(d - self.mean) / self.std for d in data_list]

    def inverse_transform(self, data_tensor):
        """
        Inverse-standardization function used when the attacker or observer needs to compute MSE
        on the original scale, such as true temperature or traffic values.
        """
        device = data_tensor.device
        dtype = data_tensor.dtype
        mean_t = torch.tensor(self.mean, device=device, dtype=dtype)
        std_t = torch.tensor(self.std, device=device, dtype=dtype)
        return (data_tensor * std_t) + mean_t

# ==========================================
# 2. PyTorch Dataset (sliding-window factory)
# ==========================================
class TSDataset(Dataset):
    def __init__(self, data_list, seq_len, pred_len, flag='test'):
        """
        Args:
            data_list: A list of time-series arrays such as [array(L, F), array(L, F), ...].
            flag: One of 'train', 'val', or 'test'.
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data_list = data_list
        self.valid_indices = []  # Store [(file index, start position), ...]

        type_map = {'train': 0, 'val': 1, 'test': 2}
        set_type = type_map[flag]

        for i, data in enumerate(data_list):
            length = len(data)
            num_train = int(length * 0.6)
            num_test = int(length * 0.2)
            num_vali = length - num_train - num_test

            border1s = [0, num_train - seq_len, length - num_test - seq_len]
            border2s = [num_train, num_train + num_vali, length]

            border1 = border1s[set_type]
            border2 = border2s[set_type]

            # Skip files that are too short
            if border2 - border1 < seq_len + pred_len:
                continue

            valid_len = (border2 - border1) - seq_len - pred_len + 1
            for j in range(valid_len):
                self.valid_indices.append((i, border1 + j))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        file_idx, start_pos = self.valid_indices[idx]
        data = self.data_list[file_idx]

        s_begin = start_pos
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len

        seq_x = data[s_begin:s_end]
        seq_y = data[s_end:r_end]

        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)

# ==========================================
# 3. Unified API for obtaining the DataLoader
# ==========================================
def get_dataloader_and_scaler(data_path, seq_len=96, pred_len=48, batch_size=32, flag='test'):
    """
    Read file or folder data, standardize it, and return a DataLoader for the selected split (train/val/test).
    """
    raw_data_list = []

    if os.path.isfile(data_path):
        # Single file (for example, ETTh1.csv).
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            data = df.select_dtypes(include=[np.number]).values
            raw_data_list.append(data)
    elif os.path.isdir(data_path):
        # Directory input (for example, UTSD-full-npy)
        # Search for all .npy files and load them in numeric order (0.npy, 1.npy, ...)
        files = glob.glob(os.path.join(data_path, "**/*.npy"), recursive=True)
        
        def sort_key(f):
            name = os.path.basename(f).split('.')[0]
            return int(name) if name.isdigit() else name
            
        files = sorted(files, key=sort_key)
        for fp in files:
            data = np.load(fp)
            if data.ndim == 1:
                data = data.reshape(-1, 1)  # Convert 1D arrays to 2D (L, 1)
            raw_data_list.append(data)
    
    if not raw_data_list:
        raise ValueError(f"No data found: {data_path}")


    scaler = StandardScaler()
    train_chunks = []
    for data in raw_data_list:
        train_len = int(len(data) * 0.6)
        train_chunks.append(data[:train_len])
        
    scaler.fit(train_chunks)
    scaled_data_list = scaler.transform(raw_data_list)


    dataset = TSDataset(scaled_data_list, seq_len, pred_len, flag=flag)
    
    shuffle = (flag == 'train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    return dataloader, scaler
