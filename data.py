from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data_frame, mode):
        self.data_frame = data_frame

        self._transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std)
        ])

    def __getitem__(self, idx):
        [f_name, labels] = self.data_frame.iloc[idx]

    def __len__(self):
        return len(self.data_frame)
