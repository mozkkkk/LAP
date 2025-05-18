import torch.nn as nn
import torch.nn.functional as F
import torch

from .Nam import NAM
from einops import rearrange
class Adapter(nn.Module):
    def __init__(self, in_features=256):
        super().__init__()
        self.ad = Mona(in_features)
        self.ps = [8,8]
        self.conv1 = nn.Conv2d(self.ps[0]*2,self.ps[0]*2,kernel_size=1)
        self.conv2 = nn.Conv2d(self.ps[1]*2,self.ps[1]*2,kernel_size=1)
        self.conv3 = nn.Conv2d(in_features,in_features,kernel_size=1)

    def forward(self, x1):
        x1,shape,pad = patch_(x1,self.ps)
        h, w = shape[2:]
        ph, pw = pad
        hps = (h + ph) // self.ps[0]
        wps = (w + pw) // self.ps[1]
        x1 = self.ad(x1,self.ps)
        if wps>1 and hps>1:
            p_l = rearrange(
                x1,
                "(b h_patches w_patches_half mul) (p_h p_w) c -> (b h_patches w_patches_half) (p_w mul) p_h c",
                b=shape[0],h_patches=hps, p_h=self.ps[0], p_w=self.ps[1], w_patches_half=wps//2, mul=2 # 固定 patch size
            )

            p_r = rearrange(
                x1,
                "(b h_patches_half mul w_patches) (p_h p_w) c -> (b h_patches_half w_patches) (p_h mul) c p_w",
                b=shape[0], p_h=self.ps[0], p_w=self.ps[1], w_patches=wps, h_patches_half=hps//2, mul=2
            )
            p_l = self.conv1(p_l)
            p_r = self.conv2(p_r)
            p_l = rearrange(
                p_l,
                "(b h_patches w_patches_half) (p_w mul) p_h c -> (b h_patches w_patches_half mul) (p_h p_w) c",
                b=shape[0],h_patches=hps, p_h=self.ps[0], p_w=self.ps[1], w_patches_half=wps//2, mul=2
            )

            p_r = rearrange(
                p_r,
                "(b h_patches_half w_patches) (p_h mul) c p_w -> (b h_patches_half mul w_patches) (p_h p_w) c",
                b=shape[0], p_h=self.ps[0], p_w=self.ps[1], w_patches=wps, h_patches_half=hps // 2, mul=2 # 固定 patch size
            )
            #x1 = self.efficient_spatial_mixing(x1, shape[0], (shape[2]+pad[0])//8, (shape[3]+pad[1])//8)
            x2 = (p_l + p_r)/2
            x2 = merge_(x2, shape, pad)
            x2 = self.conv3(x2)

        x1 = merge_(x1,shape,pad)
        if wps>1 and hps>1:
            x1 = (x1 + x2)/2


        return x1


def patch_(feat,ps):
    shape = feat.shape
    b, c, h, w = shape

    # 计算 8x8 对齐的 padding
    ph = (ps[0] - h % ps[0]) % ps[0]
    pw = (ps[1] - w % ps[1]) % ps[1]

    # 应用 padding
    if ph > 0:
        feat = F.pad(feat, (0, 0, 0, ph), mode='constant', value=0)
    if pw > 0:
        feat = F.pad(feat, (0, pw, 0, 0), mode='constant', value=0)

    # 使用单次 rearrange 替代所有 reshape/permute
    feat = rearrange(
        feat,
        "b c (h_patches p_h) (w_patches p_w) -> (b h_patches w_patches) (p_h p_w) c",
        p_h=8, p_w=8  # 固定 patch size
    )



    return feat, shape, [ph, pw]


def merge_(feat, shape=None, pad=[0, 0]):
    b, c, h, w = shape
    ph, pw = pad
    h_padded = h + ph
    w_padded = w + pw

    # 逆向变换
    feat = rearrange(
        feat,
        "(b h_patches w_patches) (p_h p_w) c -> b c (h_patches p_h) (w_patches p_w)",
        p_h=8, p_w=8,
        h_patches=h_padded // 8,
        w_patches=w_padded // 8,
        b=b
    )

    # 裁剪 padding
    return feat[:, :, :h, :w]


class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = NAM(in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)

        x = conv1_x  + identity

        identity = x

        x = self.projector(x)

        return identity + x

class Mona(nn.Module):
    def __init__(self,
                 in_dim,
                 factor=4):
        super().__init__()
        
        INNER_DIM = 64
        self.project1 = nn.Linear(in_dim, INNER_DIM)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(INNER_DIM, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        self.adapter_conv = MonaOp(INNER_DIM)

        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        identity = x

        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)

        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2 