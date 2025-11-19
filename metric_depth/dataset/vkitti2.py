import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop
from depth_anything_v2.radar import load_radar_mat
import os


class VKITTI2(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518)):
        
        self.mode = mode
        self.size = size
        
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))
    
    def __getitem__(self, item):
        parts = self.filelist[item].split(' ')
        img_path = parts[0]
        depth_path = parts[1]
        radar_path = parts[2] if len(parts) > 2 and parts[2] not in ('-', 'None', '') else None
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100.0  # cm to m
        
        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        
        sample['valid_mask'] = (sample['depth'] <= 80)
        
        sample['image_path'] = parts[0]
        # optionally load radar if present
        if radar_path:
            try:
                rp = radar_path if os.path.isabs(radar_path) else os.path.join(os.path.dirname(os.path.dirname(__file__)), radar_path)
                rd_map, rm_mask = load_radar_mat(rp, resize_to=(self.size[1], self.size[0]))
                sample['radar_depth'] = torch.from_numpy(rd_map).unsqueeze(0).float()
                sample['radar_mask'] = torch.from_numpy(rm_mask).unsqueeze(0).float()
            except Exception:
                sample['radar_depth'] = torch.zeros((1, self.size[1], self.size[0]), dtype=torch.float32)
                sample['radar_mask'] = torch.zeros((1, self.size[1], self.size[0]), dtype=torch.float32)
        else:
            sample['radar_depth'] = torch.zeros((1, self.size[1], self.size[0]), dtype=torch.float32)
            sample['radar_mask'] = torch.zeros((1, self.size[1], self.size[0]), dtype=torch.float32)
        
        return sample

    def __len__(self):
        return len(self.filelist)