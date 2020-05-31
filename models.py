import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.modules import Flatten


class Lemon(nn.Module):
    '''
    A half-assed squeeze module

    Arguments
    ----------
    'in_channels' = an `Integer`: number of input channels
    'out_channels' = an `Integer`: number of output channels
    'squeeze_factor' = an `Integer`: mid_chan = out_chan // squeeze_factor
    'kernel_size' = an `Integer`: size of the kernel expand convolution
    'stride' = an `Integer`: length of the strides for the expand convolution
    'padding' = a `Integer`: padding of the expand convolution
    'p' = a `Float`: dropout probability

    Returns
    -------
    'output' = A `Tensor` with the same type as `input`
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 squeeze_factor=1,
                 stride=1,
                 padding=0,
                 p=0.5):
        super(Lemon, self).__init__()

        # Lemon Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.squeeze_factor = squeeze_factor
        mid_channels = out_channels // squeeze_factor
        self.mid_channels = mid_channels
        self.stride = stride
        self.padding = padding

        # Squeeze convolution
        self.Squeeze = nn.Conv2d(in_channels, mid_channels, 1)
        self.SqueezeBn = nn.BatchNorm2d(mid_channels)
        self.SqueezeReLU = nn.ReLU()

        # Expand convolution
        self.Expand = nn.Conv2d(mid_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                groups=mid_channels)
        self.ExpandBn = nn.BatchNorm2d(out_channels)
        self.ExpandReLU = nn.ReLU()
        self.ExpandDropout = nn.Dropout2d(p=p)

    def forward(self, x):

        # Squeeze
        x = self.Squeeze(x)
        x = self.SqueezeBn(x)
        x = self.SqueezeReLU(x)

        # Expand
        x = self.Expand(x)
        x = self.ExpandBn(x)
        x = self.ExpandReLU(x)
        x = self.ExpandDropout(x)
        return x


class MoNetizeBody_old(nn.Module):
    def __init__(self, C=200, squeeze_factor=2, p=0.5, downsample=0):
        super(MoNetizeBody, self).__init__()

        # downsample state
        self.downsample = downsample

        # Preliminary layer
        self.PreConv = nn.Conv2d(3,
                                 48,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 groups=3)
        self.PreConvBn = nn.BatchNorm2d(48)
        self.PreConvReLU = nn.ReLU()

        # Pre downsample
        if self.downsample == 1:
            self.PreDownsample = nn.MaxPool2d(3, 2)
        elif self.downsample == 2:
            self.PreDownsample = nn.AvgPool2d(3, 2)
        else:
            self.PreDownsample = nn.MaxPool2d(3, 2)

        # Lemonade 1.1
        self.Lemon1_1 = Lemon(48,
                              64,
                              kernel_size=3,
                              squeeze_factor=squeeze_factor,
                              stride=1,
                              padding=1,
                              p=p)

        # Lemonade 1.2
        self.Lemon1_2 = Lemon(64,
                              64,
                              kernel_size=3,
                              squeeze_factor=squeeze_factor,
                              stride=1,
                              padding=1,
                              p=p)

        # lemonade 1.3: downsample
        if self.downsample == 1:
            self.Lemon1_3 = nn.MaxPool2d(3, 2)
        elif self.downsample == 2:
            self.Lemon1_3 = nn.AvgPool2d(3, 2)
        else:
            self.Lemon1_3 = nn.MaxPool2d(3, 2)

        # Lemonade 2.1
        self.Lemon2_1 = Lemon(64,
                              128,
                              kernel_size=3,
                              squeeze_factor=squeeze_factor,
                              stride=1,
                              padding=1,
                              p=p)

        # Lemonade 2.2
        self.Lemon2_2 = Lemon(128,
                              128,
                              kernel_size=3,
                              squeeze_factor=squeeze_factor,
                              stride=1,
                              padding=1,
                              p=p)

        # Lemonade 2.3: downsample
        if self.downsample == 1:
            self.Lemon2_3 = nn.MaxPool2d(3, 2)
        elif self.downsample == 2:
            self.Lemon2_3 = nn.AvgPool2d(3, 2)
        else:
            self.Lemon2_3 = nn.MaxPool2d(3, 2)

        # Lemonade 3.1
        self.Lemon3_1 = Lemon(128,
                              256,
                              kernel_size=3,
                              squeeze_factor=squeeze_factor,
                              stride=1,
                              padding=1,
                              p=p)

        # Lemonade 3.2
        self.Lemon3_2 = Lemon(256,
                              256,
                              kernel_size=3,
                              squeeze_factor=squeeze_factor,
                              stride=1,
                              padding=1,
                              p=p)

        # Lemonade 3.3
        self.Lemon3_3 = Lemon(256,
                              256,
                              kernel_size=3,
                              squeeze_factor=squeeze_factor,
                              stride=1,
                              padding=1,
                              p=p)

    def forward(self, x):

        # Preliminary convolution
        x = self.PreConv(x)
        x = self.PreConvBn(x)
        x = self.PreConvReLU(x)
        if self.downsample == 0:
            x = F.interpolate(x, scale_factor=0.2, mode='bicubic')
        else:
            x = self.PreDownsample(x)

        # First tray of lemonades
        x = self.Lemon1_1(x)
        x = self.Lemon1_2(x)
        if self.downsample == 0:
            x = F.interpolate(x, scale_factor=0.7, mode='bicubic')
        else:
            x = self.Lemon1_3(x)

        # Second tray of lemonades
        x = self.Lemon2_1(x)
        x = self.Lemon2_2(x)
        if self.downsample == 0:
            x = F.interpolate(x, scale_factor=0.9, mode='bicubic')
        else:
            x = self.Lemon2_3(x)

        # Third tray of lemonades
        x = self.Lemon3_1(x)
        x = self.Lemon3_2(x)
        x = self.Lemon3_3(x)
        return x


class MoNetizeHead_old(nn.Module):
    def __init__(self, C=200):
        super(MoNetizeHead, self).__init__()

        # Final layers
        self.FinalAvgPool = nn.AdaptiveMaxPool2d(1)
        self.FinalFlatten = Flatten()
        self.FinalLinear = nn.Linear(256, C)
        self.FinalSoftmax = nn.Softmax(1)

    def forward(self, x):

        # Exit strategy
        x = self.FinalAvgPool(x)
        x = self.FinalFlatten(x)
        x = self.FinalLinear(x)
        x = self.FinalSoftmax(x)
        return x


class MoNetize_old(nn.Module):
    def __init__(self, C=200, squeeze_factor=2, p=0.5, downsample=0):
        super(MoNetize, self).__init__()

        # Preliminary layer: /2
        self.MoNetizeBody = MoNetizeBody_old(C=C,
                                             squeeze_factor=squeeze_factor,
                                             p=p,
                                             downsample=downsample)
        self.MoNetizeHead = MoNetizeHead_old(C=C)

    def forward(self, x):

        # MoNetize
        x = self.MoNetizeBody(x)
        x = self.MoNetizeHead(x)
        return x


class Concat(nn.Module):
    def forward(self, *xs):
        return torch.cat(xs, 1)


class ResLayer(nn.Module):
    '''
    A stripped down resnet module

    Arguments
    ----------
    'in_channels' = an `Integer`: number of input channels
    'out_channels' = an `Integer`: number of output channels
    'kernel_size' = an `Integer`: size of the kernel expand convolution
    'stride' = an `Integer`: length of the strides for the expand convolution
    'padding' = a `Integer`: padding of the expand convolution

    Returns
    -------
    'output' = A `Tensor` with the same type as `input`
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 p=0.1):
        super(ResLayer, self).__init__()

        # Lemon Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.p = p

        # Squeeze convolution
        self.Res = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                             padding)
        self.ResBn = nn.BatchNorm2d(out_channels)
        self.ResReLU = nn.ReLU()
        self.ResDropout = nn.Dropout2d(p=p)

    def forward(self, x):

        # Squeeze
        x = self.Res(x)
        x = self.ResBn(x)
        x = self.ResReLU(x)
        x = self.ResDropout(x)
        return x


class VigNette_old(nn.Module):
    def __init__(self, C=200, p=0.1):
        super(VigNette, self).__init__()

        # Preliminary layer
        self.PreConv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.PreConvBn = nn.BatchNorm2d(64)
        self.PreConvReLU = nn.ReLU()

        # Layers
        self.Res_1 = ResLayer(64, 128, 3, stride=2, padding=1, p=0.1)
        self.Res_2 = ResLayer(128, 256, 3, stride=2, padding=1, p=0.1)
        self.Res_3 = ResLayer(256, 256, 3, stride=2, padding=1, p=0.1)

        # Final layers
        self.FinalMaxPool = nn.MaxPool2d(kernel_size=4,
                                         stride=4,
                                         padding=0,
                                         dilation=1,
                                         ceil_mode=False)
        self.FinalAvgPool = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        self.FinalConcat = Concat()
        self.FinalFlatten = Flatten()
        self.FinalLinear = nn.Linear(512, C)
        self.FinalSoftmax = nn.Softmax(1)

    def forward(self, x):

        # Preliminary layer
        x = self.PreConv(x)
        x = self.PreConvBn(x)
        x = self.PreConvReLU(x)

        # Resnet layers
        x = self.Res_1(x)
        x = self.Res_2(x)
        x = self.Res_3(x)

        # Pooling concatenation
        x1 = self.FinalMaxPool(x)
        x2 = self.FinalAvgPool(x)
        x = self.FinalConcat(x1, x2)

        # Final layers
        x = self.FinalFlatten(x)
        x = self.FinalLinear(x)
        x = self.FinalSoftmax(x)
        return x


class Interpolate(nn.Module):
    '''
    A class for the interpolation layers, which is only functional

    Arguments
    ----------
    'scale_factor' = a `Float`: layer scaling
    'mode' = a `String`: interpolation scheme


    Returns
    -------
    'output' = A `Tensor` with the same type as `x`
    '''
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x,
                        scale_factor=self.scale_factor,
                        mode=self.mode,
                        align_corners=False)
        return x


class Fire(nn.Module):
    '''
    Squeezenet Fire modules as in the original pytorch implementation

    Arguments
    ----------
    'inplanes' = an `Integer`: number of input channels
    'squeeze_planes' = an `Integer`: number of squeezed channels
    'expand1x1_planes' = an `Integer`: number of output 1x1 channels
    'expand3x3_planes' = an `Integer`: number of output 3x3 channels

    Returns
    -------
    'output' = A `Tensor` with the same type as `x`, concatenating the 1x1
               and the 3x3 convolution outputs
    '''
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes,
                 expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes,
                                   expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes,
                                   expand3x3_planes,
                                   kernel_size=3,
                                   padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class MoNetize(nn.Module):
    '''
    A SqueezeNet inspired NNet for Gatys style neural style transfer

    Arguments
    ----------
    'version' = a `String`: 1.0 or 1.1 squeezenet architecture
    'num_classes' = an `Integer`: number of classes

    Returns
    -------
    'output' = A `Tensor` with the same type as `x`, concatenating the 1x1
               and the 3x3 convolution outputs
    '''
    def __init__(self, version='1_0', num_classes=1000):
        super(MoNetize, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(nn.Dropout(p=0.5), final_conv,
                                        nn.ReLU(inplace=True),
                                        nn.AdaptiveAvgPool2d((1, 1)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)
