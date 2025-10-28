import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusion(nn.Module):
    """concat -> 1x1 proj -> conv -> gated fusion"""
    def __init__(self, in_ch_v, in_ch_r, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch_v + in_ch_r, out_ch, kernel_size=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch_v + in_ch_r, out_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feat_v, feat_r):
        if feat_r is None:
            return feat_v
        if feat_r.shape[-2:] != feat_v.shape[-2:]:
            feat_r = F.interpolate(feat_r, size=feat_v.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([feat_v, feat_r], dim=1)
        x = self.proj(x)
        g = self.gate(torch.cat([feat_v, feat_r], dim=1))
        out = self.out_conv(x) * g + feat_v * (1 - g)
        return out


class SEFusion(nn.Module):
    """Squeeze-and-Excitation style channel weighting fusion."""
    def __init__(self, in_ch_v, in_ch_r, reduction=16):
        super().__init__()
        self.in_ch_v = in_ch_v
        self.in_ch_r = in_ch_r
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch_v + in_ch_r, (in_ch_v + in_ch_r) // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d((in_ch_v + in_ch_r) // reduction, in_ch_v + in_ch_r, 1),
            nn.Sigmoid()
        )
        self.project = nn.Conv2d(in_ch_v + in_ch_r, in_ch_v, 1)

    def forward(self, feat_v, feat_r):
        if feat_r is None:
            return feat_v
        if feat_r.shape[-2:] != feat_v.shape[-2:]:
            feat_r = F.interpolate(feat_r, size=feat_v.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([feat_v, feat_r], dim=1)
        w = self.fc(x)
        x = x * w
        out = self.project(x)
        return out


class CrossAttentionFusion(nn.Module):
    """A lightweight cross-attention where visual features query radar tokens.
    Note: radar tokens should be flattened (B, N_r, C_r) or we will flatten inside.
    This implementation assumes radar features are small in spatial size.
    """
    def __init__(self, in_ch_v, in_ch_r, num_heads=4, dim_head=64):
        super().__init__()
        d_model = in_ch_v
        self.num_heads = num_heads
        self.to_q = nn.Conv2d(in_ch_v, d_model, 1)
        self.to_k = nn.Conv2d(in_ch_r, d_model, 1)
        self.to_v = nn.Conv2d(in_ch_r, d_model, 1)
        self.out = nn.Conv2d(d_model, in_ch_v, 1)

    def forward(self, feat_v, feat_r):
        if feat_r is None:
            return feat_v
        # align spatial sizes by downsampling visual to radar size for attention if radar is smaller
        H_v, W_v = feat_v.shape[-2:]
        H_r, W_r = feat_r.shape[-2:]
        if (H_r * W_r) <= (H_v * W_v):
            # compute q on visual (flatten), k/v on radar
            q = self.to_q(feat_v).flatten(2).permute(0, 2, 1)  # B, Nq, C
            k = self.to_k(feat_r).flatten(2).permute(0, 2, 1)  # B, Nr, C
            v = self.to_v(feat_r).flatten(2).permute(0, 2, 1)
        else:
            # downsample visual
            feat_v_ds = F.adaptive_avg_pool2d(feat_v, feat_r.shape[-2:])
            q = self.to_q(feat_v_ds).flatten(2).permute(0, 2, 1)
            k = self.to_k(feat_r).flatten(2).permute(0, 2, 1)
            v = self.to_v(feat_r).flatten(2).permute(0, 2, 1)

        # scaled dot-product attention
        d = q.shape[-1]
        attn = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        context = torch.matmul(attn, v)  # B, Nq, C

        # reshape context back to visual spatial
        if (H_r * W_r) <= (H_v * W_v):
            context = context.permute(0, 2, 1).reshape(feat_v.shape)
        else:
            context = context.permute(0, 2, 1).reshape(feat_v_ds.shape)
            context = F.interpolate(context, size=(H_v, W_v), mode='bilinear', align_corners=False)

        out = self.out(context)
        return feat_v + out
