#!/usr/bin/env python3
"""
Generate split files that include radar file paths as a third column.

Behavior:
- Read existing split files at `dataset/splits/{train,val,test}.txt` (if present).
- For each line (image_path depth_path), try to find a matching radar .mat under `dataset/radar/**`.
- Matching strategy: look for a directory or filename that contains the datetime up to seconds
  (i.e. truncate the fractional seconds). If found, pick the first .mat inside that dir.
- If no radar found, write '-' as placeholder.

Writes output to `dataset/splits_with_radar/{train,val,test}.txt` to avoid overwriting original splits.
"""
import os
import sys
import glob
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_RADAR = os.path.join(ROOT, 'dataset', 'radar')
SRC_SPLITS = os.path.join(ROOT, 'dataset', 'splits')
OUT_SPLITS = os.path.join(ROOT, 'dataset', 'splits_with_radar')

def get_timestamp_key(path):
    """
    Extracts a 'YYYY_MM_DD_HH_MM_SS' timestamp key from a file path using regex.
    """
    filename = os.path.basename(path)
    match = re.search(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})', filename)
    if match:
        return match.group(1)
    return None

def build_radar_index():
    """Walk dataset/radar once and build a map from timestamp_key -> radar_file_path."""
    index = {}
    if not os.path.isdir(DATASET_RADAR):
        return index
    
    print(f"Building radar index from: {DATASET_RADAR}")
    for root, _, files in os.walk(DATASET_RADAR):
        for file in files:
            file_path = os.path.join(root, file)
            key = get_timestamp_key(file_path)
            if key and key not in index:
                index[key] = file_path
                
    print(f"Radar index built. Found {len(index)} unique radar files.")
    return index

def find_radar_for_timestamp(img_key, index):
    # Direct hit on timestamp key
    return index.get(img_key)

def process_file(in_path, out_path, index):
    with open(in_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as fout:
        found = 0
        for l in lines:
            parts = l.split()
            if len(parts) < 2:
                continue
            img, depth = parts[0], parts[1]
            
            img_key = get_timestamp_key(img)
            
            if img_key:
                radar_match = find_radar_for_timestamp(img_key, index)
                if radar_match:
                    radar_rel = os.path.relpath(radar_match, ROOT)
                    fout.write(f"{img} {depth} {radar_rel}\n")
                    found += 1
                else:
                    fout.write(f"{img} {depth} -\n")
            else:
                fout.write(f"{img} {depth} -\n")
                
    print(f"Processed {in_path} -> {out_path}, matched radar: {found}/{len(lines)}")

def main():
    if not os.path.isdir(SRC_SPLITS):
        print(f"Source splits dir not found: {SRC_SPLITS}")
        sys.exit(1)
    os.makedirs(OUT_SPLITS, exist_ok=True)
    # build index once
    radar_index = build_radar_index()

    for name in ['train.txt', 'val.txt', 'test.txt']:
        in_p = os.path.join(SRC_SPLITS, name)
        if os.path.exists(in_p):
            out_p = os.path.join(OUT_SPLITS, name)
            process_file(in_p, out_p, radar_index)
        else:
            print(f"Skipping missing split: {in_p}")

if __name__ == '__main__':
    main()
