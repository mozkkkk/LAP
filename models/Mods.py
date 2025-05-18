import torch
import torch.nn as nn

from models.TBC import TiedBlockConv2d


class CCconv(nn.Module):
    def __init__(self, in_Channels, out_channel, ks, pad, B=8):
        super(CCconv, self).__init__()
        if in_Channels < B:
            B = 1
        self.conv1 = TiedBlockConv2d(in_Channels, out_channel, ks, padding=pad)
        self.B = B
        if self.B != 1:
            group_dim = out_channel // self.B
            # 相邻组特征拼接卷积
            self.pair_conv = nn.Conv2d(2 * group_dim, group_dim, 1)

            self.pw1 = nn.Conv2d((self.B - 1) * group_dim, out_channel, 1)

            self.all_conv = nn.Conv2d(out_channel, group_dim, 1)

            self.pw2 = nn.Conv2d(group_dim, out_channel, 1)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        if self.B != 1:
            B, C, H, W = x.shape
            group_dim = C // self.B

            # 组特征重组
            x_groups = x.view(B, self.B, group_dim, H, W)

            # 创建滑动窗口组对 (B, B-1, 2*group_dim, H, W)
            front = x_groups[:, :-1]  # 前B-1组
            back = x_groups[:, 1:]  # 后B-1组
            pairs = torch.cat([front, back], dim=2)  # 拼接相邻组

            # 并行处理所有组对
            relations1 = self.pair_conv(pairs.view(-1, 2 * group_dim, H, W))
            relations1 = relations1.view(B, (self.B - 1) * group_dim, H, W)
            relations1 = self.pw1(relations1)

            relations2 = self.all_conv(x)
            relations2 = self.pw2(relations2)
            x = (relations1 + relations2 + x) / 3

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    device = "cuda"
    cc = CCconv(256,256,1,0)
    test = cc(torch.ones([4, 256, 16, 16]).float())
    print(test.shape)

