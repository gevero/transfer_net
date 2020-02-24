import torch.nn as nn
from torch.nn.modules import Flatten


class TransferNet(nn.Module):
    def __init__(self):
        super(TransferNet, self).__init__()

        # Preliminary layer
        self.prep_conv = nn.Conv2d(3,
                                   64,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1),
                                   bias=False)
        self.prep_bn = nn.BatchNorm2d(64,
                                      eps=1e-05,
                                      momentum=0.1,
                                      affine=True,
                                      track_running_stats=True)
        self.prep_relu = nn.ReLU()

        # Layer 2
        self.layer2_conv = nn.Conv2d(64,
                                     128,
                                     kernel_size=(3, 3),
                                     stride=(1, 1),
                                     padding=(1, 1),
                                     bias=False)
        self.layer2_bn = nn.BatchNorm2d(128,
                                        eps=1e-05,
                                        momentum=0.1,
                                        affine=True,
                                        track_running_stats=True)
        self.layer2_relu = nn.ReLU()
        self.layer2_pool = nn.MaxPool2d(kernel_size=2,
                                        stride=2,
                                        padding=0,
                                        dilation=1,
                                        ceil_mode=False)

        # Layer 3
        self.layer3_conv = nn.Conv2d(128,
                                     256,
                                     kernel_size=(3, 3),
                                     stride=(1, 1),
                                     padding=(1, 1),
                                     bias=False)
        self.layer3_bn = nn.BatchNorm2d(256,
                                        eps=1e-05,
                                        momentum=0.1,
                                        affine=True,
                                        track_running_stats=True)
        self.layer3_relu = nn.ReLU()
        self.layer3_pool = nn.MaxPool2d(kernel_size=2,
                                        stride=2,
                                        padding=0,
                                        dilation=1,
                                        ceil_mode=False)

        # Layer 4
        self.layer4_conv = nn.Conv2d(256,
                                     512,
                                     kernel_size=(3, 3),
                                     stride=(1, 1),
                                     padding=(1, 1),
                                     bias=False)
        self.layer4_bn = nn.BatchNorm2d(512,
                                        eps=1e-05,
                                        momentum=0.1,
                                        affine=True,
                                        track_running_stats=True)
        self.layer4_relu = nn.ReLU(inplace=True)
        self.layer4_pool = nn.MaxPool2d(kernel_size=2,
                                        stride=2,
                                        padding=0,
                                        dilation=1,
                                        ceil_mode=False)

        # Final layer
        self.final_pool = nn.MaxPool2d(kernel_size=4,
                                       stride=4,
                                       padding=0,
                                       dilation=1,
                                       ceil_mode=False)
        self.final_flatten = Flatten()
        self.final_linear = nn.Linear(in_features=512,
                                      out_features=10,
                                      bias=True)

    def forward(self, x):

        # Preliminary layer
        x = self.prep_conv(x)
        x = self.prep_bn(x)
        x = self.prep_relu(x)

        # Layer 2
        x = self.layer2_conv(x)
        x = self.layer2_bn(x)
        x = self.layer2_relu(x)
        x = self.layer2_pool(x)

        # Layer 3
        x = self.layer3_conv(x)
        x = self.layer3_bn(x)
        x = self.layer3_relu(x)
        x = self.layer3_pool(x)

        # Layer 4
        x = self.layer4_conv(x)
        x = self.layer4_bn(x)
        x = self.layer4_relu(x)
        x = self.layer4_pool(x)

        # Final layer
        x = self.final_pool(x)
        x = self.final_flatten(x)
        x = self.final_linear(x)
        return x
