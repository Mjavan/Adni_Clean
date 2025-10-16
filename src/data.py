import torch
import numpy as np
import pandas as pd
import os

import torchvision
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import nibabel as nib

from pathlib import Path
import matplotlib.pyplot as plt


class AdniMRIDataset2D(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir, self.img_labels["archive_fname"].iloc[idx])
        # print(img_path)
        image = self.read_image(img_path)
        label = self.img_labels["group"].iloc[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def read_image(self, path):
        img = nib.load(path).get_fdata().astype(np.uint8)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]  # HW -> HWC
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return img