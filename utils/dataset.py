import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CelebFaceDataset(Dataset):
    def __init__(self, image_dir, size=None, transform=None):
        self.image_dir = image_dir
        all_files = sorted([
            f for f in os.listdir(image_dir) if f.endswith('.jpg')
        ])
        self.image_files = all_files[:size] if size is not None else all_files
        
        self.transform = transform if transform is not None else  transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, IOError):
            image = Image.new('RGB', (64, 64), color=0)
        return self.transform(image), idx