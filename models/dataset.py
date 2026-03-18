import os
import torch
import numpy as np
import torch.utils.data as data


class mesh_pc_dataset(data.Dataset):
    def __init__(self, data_path, mode, num_sample):
        super(mesh_pc_dataset, self).__init__()
        self.data_path = data_path
        self.mode = mode
        assert mode in ['train', 'test']
        self.num_sample = num_sample
        
        self.npz_files = [f for f in os.listdir(data_path) if f.endswith('.npz')]
        self.npz_path = [os.path.join(data_path, f) for f in self.npz_files]

    def __len__(self):
        return len(self.npz_path)

    def __getitem__(self, idx):
        npz_data = np.load(self.npz_path[idx])

        sample_pc = npz_data['samples'] # (N, 3)
        closest_points = npz_data['closest_points'] # (N, 3)
        points_gt = npz_data['points']  # (M, 3)

        idx = np.random.choice(np.arange(sample_pc.shape[0]), self.num_sample)

        sample_pc = sample_pc[idx, :]
        closest_points = closest_points[idx, :]

        return {
            'sample_pc': torch.from_numpy(sample_pc).float(),   # (N, 3)
            'closest_points': torch.from_numpy(closest_points).float(), # (N, 3)
            'points_gt': torch.from_numpy(points_gt).float(),   # (M, 3)
        }
