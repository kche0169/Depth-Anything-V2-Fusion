
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
import argparse

# We need to add the project root to the path to import the dataset modules
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metric_depth.dataset.hypersim import Hypersim
from metric_depth.dataset.kitti import KITTIDataset
from metric_depth.dataset.vkitti2 import VKITTI2Dataset

DATASETS = {
    'hypersim': Hypersim,
    'kitti': KITTIDataset,
    'vkitti2': VKITTI2Dataset
}

def test_dataset(dataset_name, split_file, batch_size=2, num_workers=0, num_samples=5):
    print(f"--- Testing dataset: {dataset_name} ---")
    print(f"Split file: {split_file}")

    if not os.path.exists(split_file):
        print(f"Error: Split file not found at {split_file}")
        return

    try:
        dataset_class = DATASETS[dataset_name]
        dataset = dataset_class(split_file, height=352, width=704)
    except KeyError:
        print(f"Error: Dataset '{dataset_name}' not found.")
        return
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    if len(dataset) == 0:
        print("Error: Dataset is empty. Check the split file and data paths.")
        return
        
    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    print(f"Fetching {num_samples} samples...")
    for i, sample in enumerate(dataloader):
        if i >= num_samples:
            break
        
        print(f"\n--- Sample {i+1} ---")
        print("Keys:", sample.keys())

        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: tensor, shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  - {key}: {type(value)}")

        if 'radar_depth' in sample and 'radar_mask' in sample:
            radar_depth = sample['radar_depth']
            radar_mask = sample['radar_mask']
            
            print(f"  - radar_depth non-zero elements: {torch.count_nonzero(radar_depth)}")
            print(f"  - radar_mask non-zero elements: {torch.count_nonzero(radar_mask)}")
            
            if torch.count_nonzero(radar_mask) > 0:
                print("  => SUCCESS: Radar data found and loaded.")
            else:
                print("  => NOTE: No radar data for this sample (mask is all zeros).")
        else:
            print("  => WARNING: 'radar_depth' or 'radar_mask' not in sample.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test data loading for depth estimation datasets.")
    parser.add_argument('--dataset', type=str, default='hypersim', choices=DATASETS.keys(), help='Dataset to test.')
    parser.add_argument('--split', type=str, default='train', help='Split to test (e.g., train, val).')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to inspect.')
    
    args = parser.parse_args()

    # Assuming the new splits are in 'dataset/splits_with_radar/'
    split_file_path = os.path.join('dataset', 'splits_with_radar', args.dataset, f'{args.split}.txt')
    
    test_dataset(args.dataset, split_file_path, num_samples=args.num_samples)
