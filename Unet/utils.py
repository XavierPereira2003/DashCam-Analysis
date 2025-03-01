
import re
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class IDD_Lite(Dataset):
    def __init__(self, type="train"):
        def convert_path(p):
            return p.replace('Image', 'label').replace('_image.jpg', '_label.png')
        super().__init__()

        self.image_dirs = glob(f'idd20k_lite/Image/{type}/*/*_image.jpg')
        self.mask_dirs = [convert_path(p) for p in self.image_dirs]

    def __len__(self):
        return len(self.image_dirs)
    
    def __getitem__(self, index):
        image = Image.open(self.image_dirs[index])
        mask = Image.open(self.mask_dirs[index])
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])
        image = transform(image)
        mask = transform(mask)
        return {"image":image, "mask":mask}

    def getDirs(self, idx):
        return self.image_dirs[idx], self.mask_dirs[idx]

