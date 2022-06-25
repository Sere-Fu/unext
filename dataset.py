import os
import cv2
import numpy as np
import torch
import torch.utils.data
from glob import glob
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir 
        self.p_ids = [os.path.basename(p).split('.')[0] for p in glob(os.path.join(data_dir, '*.jpg'))]
        self.transform = transform

    def __len__(self):
        return len(self.p_ids)

    def __getitem__(self, idx):
        p_id = self.p_ids[idx]
        
        img = cv2.imread(os.path.join(self.data_dir, f"{p_id}.jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        mask = np.asarray(Image.open(os.path.join(self.data_dir, f"{p_id}.png")))

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.transpose(2, 0, 1)
        mask = np.expand_dims(mask, 0).astype(np.float32)

        return img, mask
