import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop
from depth_anything_v2.radar import load_radar_mat
import os


def hypersim_distance_to_depth(npyDistance):
    intWidth, intHeight, fltFocal = 1024, 768, 886.81

    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(
        1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5,
                                 intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate(
        [npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
    return npyDepth


class Hypersim(Dataset):
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
        
        depth_fd = h5py.File(depth_path, "r")
        distance_meters = np.array(depth_fd['dataset'])
        depth = hypersim_distance_to_depth(distance_meters)
        
        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        
        sample['valid_mask'] = (torch.isnan(sample['depth']) == 0)
        sample['depth'][sample['valid_mask'] == 0] = 0
        
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