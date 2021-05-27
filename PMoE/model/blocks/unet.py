"""Basic implementation of U-Net with cross channel co-efficients"""
import torch
import torch.nn as nn

from .basics import conv3, EfficientBlock


class UNet(nn.Module):
    """Basic U-Net"""

    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 23,
        gamma: int = 2,
        b: int = 1,
        dropout: float = 0.0,
        inter_repr: bool = False,
    ):
        super().__init__()
        self.inter_repr = inter_repr
        # Contracting path
        self.dwn_1 = conv3(in_features, 64)
        self.dwn_2 = conv3(64, 128)
        self.dwn_3 = conv3(128, 256)
        self.dwn_4 = conv3(256, 512)
        self.dwn_5 = conv3(512, 512)
        # self.eca_0 = EfficientBlock(512, gamma, b)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=dropout)

        # expansive path
        self.up_1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        # self.eca_1 = EfficientBlock(512, gamma, b)
        self.up_forw_1 = conv3(1024, 512)
        self.up_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # self.eca_2 = EfficientBlock(256, gamma, b)
        self.up_forw_2 = conv3(512, 256)
        self.up_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # self.eca_3 = EfficientBlock(128, gamma, b)
        self.up_forw_3 = conv3(256, 128)
        self.up_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # self.eca_4 = EfficientBlock(64, gamma, b)
        self.up_forw_4 = conv3(128, 64)

        # out layer
        self.out = nn.Conv2d(64, out_features, kernel_size=1)

    def forward(self, image):
        # Contracting path
        x_1 = self.dwn_1(image)
        x_1 = self.dropout(x_1)
        x_2 = self.pool(x_1)

        x_2 = self.dwn_2(x_2)
        x_2 = self.dropout(x_2)
        x_3 = self.pool(x_2)

        x_3 = self.dwn_3(x_3)
        x_3 = self.dropout(x_3)
        x_4 = self.pool(x_3)

        x_4 = self.dwn_4(x_4)
        x_4 = self.dropout(x_4)
        x_5 = self.pool(x_4)

        # x_5 = self.eca_0(x_5)
        x_5 = self.dwn_5(x_5)

        # expansive path
        x = self.up_1(x_5, output_size=x_4.size())
        x = torch.cat([x_4, x], 1)
        x = self.up_forw_1(x)

        x = self.up_2(x, output_size=x_3.size())
        x = torch.cat([x_3, x], 1)
        x = self.up_forw_2(x)

        x = self.up_3(x, output_size=x_2.size())
        x = torch.cat([x_2, x], 1)
        x = self.up_forw_3(x)

        x = self.up_4(x, output_size=x_1.size())
        x = torch.cat([x_1, x], 1)
        x = self.up_forw_4(x)

        x = self.out(x)

        if self.inter_repr:
            x_5 = self.avgpool(x_5)
            x_5 = torch.flatten(x_5, 1)
            return x_5, x

        return x


class UNetECA(nn.Module):
    """U-Net with ECA coefficients"""

    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 23,
        gamma: int = 2,
        b: int = 1,
        dropout: float = 0.0,
        inter_repr: bool = False,
    ):
        super().__init__()
        self.inter_repr = inter_repr
        # Contracting path
        self.dwn_1 = conv3(in_features, 32)
        self.dwn_2 = conv3(32, 64)
        self.dwn_3 = conv3(64, 128)
        self.dwn_4 = conv3(128, 256)
        self.dwn_5 = conv3(256, 512)
        self.eca_0 = EfficientBlock(512, gamma, b)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=dropout)

        # expansive path
        self.up_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.eca_1 = EfficientBlock(512, gamma, b)
        self.up_forw_1 = conv3(512, 256)
        self.up_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.eca_2 = EfficientBlock(256, gamma, b)
        self.up_forw_2 = conv3(256, 128)
        self.up_3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.eca_3 = EfficientBlock(128, gamma, b)
        self.up_forw_3 = conv3(128, 64)
        self.up_4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.eca_4 = EfficientBlock(64, gamma, b)
        self.up_forw_4 = conv3(64, 32)

        # out layer
        self.out = nn.Conv2d(32, out_features, kernel_size=1)

    def forward(self, image):
        # Contracting path
        x_1 = self.dwn_1(image)
        x_1 = self.dropout(x_1)
        x_2 = self.pool(x_1)

        x_2 = self.dwn_2(x_2)
        x_2 = self.dropout(x_2)
        x_3 = self.pool(x_2)

        x_3 = self.dwn_3(x_3)
        x_3 = self.dropout(x_3)
        x_4 = self.pool(x_3)

        x_4 = self.dwn_4(x_4)
        x_4 = self.dropout(x_4)
        x_5 = self.pool(x_4)

        x_5 = self.eca_0(x_5)
        x_5 = self.dwn_5(x_5)

        # expansive path
        x = self.up_1(x_5, output_size=x_4.size())
        x = torch.cat([x_4, x], 1)
        x = self.up_forw_1(self.eca_1(x))

        x = self.up_2(x, output_size=x_3.size())
        x = torch.cat([x_3, x], 1)
        x = self.up_forw_2(self.eca_2(x))

        x = self.up_3(x, output_size=x_2.size())
        x = torch.cat([x_2, x], 1)
        x = self.up_forw_3(self.eca_3(x))

        x = self.up_4(x, output_size=x_1.size())
        x = torch.cat([x_1, x], 1)
        x = self.up_forw_4(self.eca_4(x))

        x = self.out(x)

        if self.inter_repr:
            x_5 = self.avgpool(x_5)
            x_5 = torch.flatten(x_5, 1)
            return x_5, x

        return x
