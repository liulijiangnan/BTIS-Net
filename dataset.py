import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class BRATSDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = [sample for sample in os.listdir(data_dir) if sample.startswith('BraTS')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        sample_dir = os.path.join(self.data_dir, sample_name)

        # Load MRI modalities and segmentation mask
        t1_path = os.path.join(sample_dir, sample_name + '_t1.nii.gz')
        t1ce_path = os.path.join(sample_dir, sample_name + '_t1ce.nii.gz')
        t2_path = os.path.join(sample_dir, sample_name + '_t2.nii.gz')
        flair_path = os.path.join(sample_dir, sample_name + '_flair.nii.gz')
        seg_path = os.path.join(sample_dir, sample_name + '_seg.nii.gz')

        h = 128
        w = 128
        t1_data = nib.load(t1_path).get_fdata()
        t1_data = np.resize(t1_data, (h, w, 155))

        t1ce_data = nib.load(t1ce_path).get_fdata()
        t1ce_data = np.resize(t1ce_data, (h, w, 155))

        t2_data = nib.load(t2_path).get_fdata()
        t2_data = np.resize(t2_data, (h, w, 155))

        flair_data = nib.load(flair_path).get_fdata()
        flair_data = np.resize(flair_data, (h, w, 155))

        seg_data = nib.load(seg_path).get_fdata()
        seg_data = seg_data.astype(np.float32)
        seg_data = np.resize(seg_data, (h, w, 155))
        seg_data = np.transpose(seg_data, (2, 0, 1))

        x = np.stack([t1_data, t1ce_data, t2_data, flair_data], axis=-1)
        x = np.transpose(x, (3, 2, 0, 1))

        # Convert numpy arrays to PyTorch tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(seg_data, dtype=torch.long)

        return x, y
