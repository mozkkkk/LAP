import torch.nn as nn
import torch.nn.functional as F
import torch
from .Nam import NAM

class FAdapter(nn.Module):
    def __init__(self, in_features1=256, in_features2=256):
        super().__init__()
        self.ad1 = Mona(in_features1)        
        self.ad2 = Mona(in_features2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x1, x2):
        x1,shape,pad = patch_(x1)
        x1 = self.ad1(x1,[16,16])
        x1 = merge_(x1,shape,pad)

        x2,shape,pad = patch_(x2)
        x2 = self.ad2(x2,[16,16])
        x2 = merge_(x2,shape,pad)

        x2 = self.down(x2) + x1

        return x2

class Adapter(nn.Module):
    def __init__(self, in_features=256):
        super().__init__()
        self.ad = Mona(in_features)        

    def forward(self, x1):
        x1,shape,pad = patch_(x1)
        x1 = self.ad(x1,[8,8])
        #x1 = self.efficient_spatial_mixing(x1, shape[0], (shape[2]+pad[0])//8, (shape[3]+pad[1])//8)
        x1 = merge_(x1,shape,pad)

        return x1


def patch_(feat):
    shape = feat.shape
    b,c,h,w = shape
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    if ph > 0:
        feat = F.pad(feat, (0, 0, 0, ph), mode='constant', value= 0)
    if pw > 0:
        feat = F.pad(feat, (0, pw, 0, 0), mode='constant', value= 0)
    feat = feat.reshape(b,c,-1,8,8)
    feat = feat.permute(0,2,1,3,4)
    feat = feat.reshape(-1,c,64)
    feat = feat.permute(0,2,1)
    return feat,shape,[ph,pw]

def merge_(feat,shape=None,pad=[0,0]):
    b,c,h,w = shape
    ph, pw = pad
    feat = feat.permute(0,2,1)
    feat = feat.reshape(-1,c,8,8)
    feat = feat.reshape(b,-1,c,8,8)
    feat = feat.permute(0,2,1,3,4)
    feat = feat.reshape(b, c, h+ph, w+pw)
    feat = feat[:,:,:h,:w]
    return feat

class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)
        self.conv4 = NAM(in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)
        conv4_x = self.conv4(x)

        x = (conv1_x + conv2_x + conv3_x + conv4_x) / 4.0 + identity

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