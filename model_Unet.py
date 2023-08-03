# -*- coding: utf-8 -*-
"""
@Time ： 11/19/2022 6:08 PM
@Auth ： YY
@File ：model_Unet.py
@IDE ：PyCharm
@state:
@Function：
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# 在U-net中,网络经常成对使用
class DoubleConv(nn.Sequential):
    # in_channels   输入特征层
    # out_channels  输出特征层
    # mid_channels  中间层,第一的
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if not mid_channels:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            # 第一对卷积
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 第二对卷积
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


# 下采样
class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )



# 上采样,拼接
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(nn.Conv2d(in_channels, num_classes, kernel_size=1))

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, bilinear=True, base_c=64):
        super(UNet, self).__init__()
        # 图片通道个数 rgb：3 灰度 ：1
        self.n_channels = n_channels
        # 分类个数 2：包括背景
        self.n_classes = n_classes
        # 上采样方法
        self.bilinear = bilinear
        # 输入层
        self.inc = DoubleConv(n_channels, base_c)
        # 第一次下采样
        self.down1 = Down(base_c, base_c * 2)
        # 第二次下采样
        self.down2 = Down(base_c * 2, base_c * 4)
        # 第三次下采样
        self.down3 = Down(base_c * 4, base_c * 8)
        # 第四次下采样
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        # 第一次上采样
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        # 第二次上采样
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        # 第三次上采样
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        # 第四次上采样
        self.up4 = Up(base_c * 2, base_c, bilinear)
        # 输出层
        self.outc = OutConv(base_c, n_classes)

    def forward(self, x):
        # 输入层
        x1 = self.inc(x)
        # 第一层下采样 输出
        x2 = self.down1(x1)
        # 第二层下采样 输出
        x3 = self.down2(x2)
        # 第三层下采样 输出
        x4 = self.down3(x3)
        # 第四层下采样 输出
        x5 = self.down4(x4)
        # 第一层上采样 输出
        x = self.up1(x5, x4)
        # 第二层上采样 输出
        x = self.up2(x, x3)
        # 第三层上采样 输出
        x = self.up3(x, x2)
        # 第四层上采样 输出
        x = self.up4(x, x1)
        # 输出层
        logits = self.outc(x)
        return logits