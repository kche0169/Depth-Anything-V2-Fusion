import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_radar_mat(mat_path, out_size=None, cam_intrinsics_path='parameters/Camera_Intrinsics.mat',
                   radar2rgb_path='parameters/Radar2RGB_Extrinsics.mat', resize_to=None):
    """
    Load a .mat radar file with 'XYZ' (3,N) and project to image pixel coordinates using camera intrinsics/extrinsics.
    Returns a sparse depth map (H,W) in meters and a mask (H,W) where 1 indicates a radar measurement.
    If resize_to is specified (h,w), the returned maps are resized to that resolution.
    """
    data = scipy.io.loadmat(mat_path)
    if 'XYZ' not in data:
        raise ValueError(f"MAT file {mat_path} does not contain 'XYZ' field")
    pts = data['XYZ']  # shape (3, N)

    # load camera intrinsics and radar->rgb extrinsics
    cam = scipy.io.loadmat(cam_intrinsics_path)['Camera_Intrinsics']
    radar2rgb = scipy.io.loadmat(radar2rgb_path)['Radar2RGB_Extrinsics']

    # pts: 3xN in radar frame -> convert to camera frame
    R = radar2rgb[:3, :3]
    t = radar2rgb[:3, 3:4]
    pts_cam = R @ pts + t  # 3xN

    z = pts_cam[2, :]
    valid = z > 0.01
    if not np.any(valid):
        # return empty maps
        if resize_to is None:
            # try to infer image size from intrinsics (not available): default 480x640
            H, W = 480, 640
        else:
            H, W = resize_to
        return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.uint8)

    pts_cam = pts_cam[:, valid]
    z = pts_cam[2, :]

    # project to pixels
    uv1 = cam @ pts_cam
    u = uv1[0, :] / uv1[2, :]
    v = uv1[1, :] / uv1[2, :]

    # infer image size: try to find from camera intrinsics principal point and assume reasonable size
    # If resize_to provided, use it
    if resize_to is not None:
        H, W = resize_to
    else:
        # fallback defaults
        H, W = 480, 640

    # round to int pixel indices
    u_pix = np.round(u).astype(int)
    v_pix = np.round(v).astype(int)

    # build sparse map
    depth_map = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.uint8)
    for ui, vi, zi in zip(u_pix, v_pix, z):
        if 0 <= vi < H and 0 <= ui < W:
            # if multiple points map to same pixel, take nearest (min z)
            if mask[vi, ui] == 0 or zi < depth_map[vi, ui]:
                depth_map[vi, ui] = float(zi)
                mask[vi, ui] = 1

    return depth_map, mask


def load_radar_file(fpath, resize_to=None):
    """
    Generic radar file loader. Assumes a simple text file format where each line is a point.
    The projection logic here is a placeholder and might need real camera intrinsics.
    """
    try:
        # This assumes a text file where columns might be separated by space or comma.
        # It tries to load points, expecting at least 3 columns (x, y, z).
        points = np.loadtxt(fpath, delimiter=',')
    except Exception:
        try:
            points = np.loadtxt(fpath) # Fallback to space delimiter
        except Exception as e:
            # print(f"Could not load radar file {fpath}: {e}")
            return _get_zero_tensors(resize_to)

    return _process_radar_points(points, resize_to)

def _process_radar_points(points, resize_to):
    """
    Projects an array of 3D points into a sparse depth map.
    This function contains placeholder camera intrinsics.
    """
    if points.ndim == 1 and points.shape[0] >= 3:
        points = np.expand_dims(points, axis=0)
    elif points.ndim != 2 or points.shape[1] < 3:
        # print(f"Invalid radar points shape: {points.shape}")
        return _get_zero_tensors(resize_to)

    h, w = resize_to if resize_to else (480, 640)
    depth_map = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.float32)

    # --- Placeholder Camera Intrinsics ---
    # These should be replaced with your actual camera parameters.
    fx, fy = 960, 960  # Focal length
    cx, cy = w / 2, h / 2  # Principal point
    # --- End Placeholder ---

    # Assuming points are in camera coordinate system (x-right, y-down, z-forward)
    for p in points:
        x, y, z = p[0], p[1], p[2]
        
        if z > 0.5:  # Filter points behind or too close to the camera
            # Project 3D point to 2D pixel
            u = int(fx * x / z + cx)
            v = int(fy * y / z + cy)

            if 0 <= u < w and 0 <= v < h:
                # If multiple points map to the same pixel, keep the one with the smallest depth
                if mask[v, u] == 0 or z < depth_map[v, u]:
                    depth_map[v, u] = z
                    mask[v, u] = 1
    
    return torch.from_numpy(depth_map).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)

def _get_zero_tensors(resize_to):
    """Returns zero tensors for depth and mask if radar loading fails."""
    h, w = resize_to if resize_to else (480, 640)
    return torch.zeros((1, h, w), dtype=torch.float32), torch.zeros((1, h, w), dtype=torch.float32)


class RadarEncoder(nn.Module):
    """Lightweight CNN encoder that returns multi-scale radar features to fuse with visual features."""
    def __init__(self, in_ch=1, features=[32, 64, 128]):
        super().__init__()
        layers = []
        prev = in_ch
        self.stages = nn.ModuleList()
        for f in features:
            self.stages.append(nn.Sequential(
                nn.Conv2d(prev, f, 3, padding=1, bias=False),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True),
                nn.Conv2d(f, f, 3, padding=1, bias=False),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True),
            ))
            prev = f
        self.projects = nn.ModuleList([nn.Conv2d(f, f, 1) for f in features])

    def forward(self, x):
        feats = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            feats.append(self.projects[i](x))
            x = F.avg_pool2d(x, 2)
        return feats
