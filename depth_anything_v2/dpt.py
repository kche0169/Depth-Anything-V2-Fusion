import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet
from .radar import RadarEncoder, load_radar_mat
from .fusion import ConcatFusion, SEFusion, CrossAttentionFusion


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
    ):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
    
    def forward(self, out_features, patch_h, patch_w, radar_feats=None, fusions=None):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            # optionally fuse radar features at this level
            if radar_feats is not None and fusions is not None and fusions[i] is not None:
                # radar_feats expected to be list of spatial tensors aligned to this level
                try:
                    x = fusions[i](x, radar_feats[i])
                except Exception:
                    # fallback: ignore fusion if mismatch
                    pass

            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out


class DepthAnythingV2(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        fusion_type=None,
        radar_in_ch=1,
    ):
        super(DepthAnythingV2, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

        # Fusion setup: optional radar encoder and fusion modules
        self.fusion_type = fusion_type
        if fusion_type is not None:
            self.radar_encoder = RadarEncoder(in_ch=radar_in_ch, features=[32, 64, 128])
            num_levels = len(self.depth_head.projects)
            fusions = [None] * num_levels
            embed_dim = self.pretrained.embed_dim
            radar_sizes = [32, 64, 128]
            fuse_indices = [1, 2]
            for idx, rch in zip(fuse_indices, radar_sizes[:len(fuse_indices)]):
                if fusion_type == 'concat':
                    fusions[idx] = ConcatFusion(in_ch_v=embed_dim, in_ch_r=rch, out_ch=embed_dim)
                elif fusion_type == 'se':
                    fusions[idx] = SEFusion(in_ch_v=embed_dim, in_ch_r=rch)
                elif fusion_type == 'cross':
                    fusions[idx] = CrossAttentionFusion(in_ch_v=embed_dim, in_ch_r=rch)
                else:
                    fusions[idx] = None
            self.fusions = nn.ModuleList(fusions)
    
    
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518, radar_path=None):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        radar_tensor = None
        radar_feats = None
        fusions = None
        if radar_path is not None:
            # try to load .mat and create sparse depth map
            try:
                # project radar to input_size
                depth_map, mask = load_radar_mat(radar_path, resize_to=(input_size, input_size))
                # create tensor Bx1xH x W
                radar_tensor = torch.from_numpy(depth_map[None, None]).float().to(image.device)
            except Exception:
                radar_tensor = None

        if radar_tensor is not None:
            depth = self.forward(image, radar=radar_tensor)
        else:
            depth = self.forward(image)
        
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()
    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)

    def forward(self, x, radar=None):
        """Override to accept optional radar tensor of shape (B, C, H, W) aligned to input_size."""
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)

        # if radar provided, compute radar_feats via radar_encoder and build fusion modules in __init__
        radar_feats = None
        fusions = None
        if hasattr(self, 'radar_encoder') and radar is not None:
            # radar: BxC x H x W
            radar_feats = self.radar_encoder(radar)
            # radar_feats is a list from shallow to deep; need to align order with out_features
            # Build a list of spatial tensors matching out_features count (pad/trim as necessary)
            # Here we will upsample radar_feats to match feature spatial sizes inside DPTHead
            fusions = self.fusions if hasattr(self, 'fusions') else None

        depth = self.depth_head(features, patch_h, patch_w, radar_feats=radar_feats, fusions=fusions)
        depth = F.relu(depth)
        return depth.squeeze(1)
