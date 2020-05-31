import torch
import torch.nn as nn
import torch.nn.init as init


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
    A SqueezeNet inspired NNet for Gatys like Neural Style Transfer

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
                # First conv + interpolate downscale
                nn.Conv2d(3, 96, kernel_size=7, stride=1),
                nn.ReLU(inplace=True),
                Interpolate(0.5, 'bilinear'),
                # Second conv + interpolate downscale
                nn.Conv2d(96, 96, kernel_size=7, stride=1),
                nn.ReLU(inplace=True),
                Interpolate(0.5, 'bilinear'),
                # First fire sequence + interpolate downscale
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                Interpolate(0.5, 'bilinear'),
                # Second fire sequence + interpolate downscale
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Interpolate(0.5, 'linear'),
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                # First conv + interpolate downscale
                nn.Conv2d(3, 64, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                Interpolate(0.5, 'bilinear'),
                # Second conv + interpolate downscale
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                Interpolate(0.5, 'bilinear'),
                # First fire sequence + interpolate downscale
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Interpolate(0.5, 'bilinear'),
                # Second fire sequence + interpolate downscale
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                Interpolate(0.5, 'bilinear'),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            # In this case it is needed
            raise ValueError("Unsupported MoNetize version {version}:"
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
