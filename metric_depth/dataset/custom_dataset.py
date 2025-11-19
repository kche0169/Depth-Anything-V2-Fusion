
import os
import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from depth_anything_v2.util.transform import Compose, Resize, NormalizeImage
from depth_anything_v2.radar import load_radar_file # We will create this function

class CustomDataset(Dataset):
    def __init__(self, split_file, height=352, width=704, **kwargs):
        super().__init__()
        
        self.height = height
        self.width = width
        
        with open(split_file, 'r') as f:
            self.lines = f.readlines()

        self.transform = Compose([
            Resize(width=self.width, height=self.height, resize_target=True, keep_aspect_ratio=False, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        parts = line.split()
        
        image_path = parts[0]
        depth_path = parts[1]
        radar_path = parts[2] if len(parts) > 2 else '-'
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load depth
        depth = np.array(Image.open(depth_path), dtype=np.float32)
        depth = np.expand_dims(depth, axis=2)
        
        # Load radar
        if radar_path != '-':
            # Use a generic radar loader which we will define
            radar_depth, radar_mask = load_radar_file(radar_path, resize_to=(self.height, self.width))
        else:
            radar_depth = torch.zeros((1, self.height, self.width), dtype=torch.float32)
            radar_mask = torch.zeros((1, self.height, self.width), dtype=torch.float32)

        sample = {'image': image, 'depth': depth, 'radar_depth': radar_depth, 'radar_mask': radar_mask}
        
        sample = self.transform(sample)
        
        # Add other necessary fields
        sample['dataset'] = 'custom'
        sample['filename'] = os.path.basename(image_path)
        
        return sample
