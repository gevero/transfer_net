#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Pytorch
import torch
import torchvision.models as models
import torch.nn.functional as F
import neural_style_utils as nsu
from collections import OrderedDict

# select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# build command line argument parser
options = nsu.build_parser().parse_args()

# load the content images
content_tensor_0 = nsu.imload(options.content, options.content_resize)
content_tensor_0 = content_tensor_0.to(device)

# initialize the stylized image
if options.random_start:
    train_tensor = torch.randn(content_tensor_0.size(), requires_grad=True)
else:
    train_tensor = content_tensor_0.clone()
    train_tensor.requires_grad = True
train_tensor = train_tensor.to(device)

# load the style image
try:
    _ = len(options.style_resize)
    style_size = options.style_resize
except:
    style_size = tuple(content_tensor_0.size()[2:])
style_tensor = [
    nsu.imload(img, style_size).to(device) for img in options.style
]
style_tensor_0 = style_tensor[0]

# mask functions
# resizefunc = transforms.Resize(style_size)
# toImage = transforms.ToPILImage()
# toTensor = transforms.ToTensor()

# create mask
# side = content_tensor.shape[-1]
# mask = torch.zeros((side, side))
# split = side // 2
# mask[:, :split] = 0
# mask[:, split:] = 1
# mask_img = toImage(mask)
# mask = toTensor(resizefunc(mask_img)).squeeze()
# mask = mask.to(device)
mask = None

# load pretrained model: whole model with classification head
model = models.squeezenet1_0(pretrained=False)

# load checkpoint
checkpoint = torch.load('./trained_models/squeezenet_imagenet.pth.tar',
                        map_location=torch.device(device))

# fix dictionary
checkpoint_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    if k[0:6] == 'module':
        name = k[7:]  # remove `module.`
        checkpoint_state_dict[name] = v

# load params
model.load_state_dict(checkpoint_state_dict)

# model to device
model.to(device)

# loop for multiscale stylization
s_array = nsu.size_array(tuple(content_tensor_0.size()[2:]),
                         options.size_steps)
for i_s, size in enumerate(s_array):

    # scale the content tensor
    content_tensor = F.interpolate(content_tensor_0, size)
    content_tensor = content_tensor.to(device)

    # scaling the style tensor
    style_tensor = F.interpolate(style_tensor_0, size)
    style_tensor = style_tensor.to(device)

    # scale the training tensor
    train_tensor = F.interpolate(train_tensor, size).detach()
    train_tensor = train_tensor.to(device)
    train_tensor.requires_grad = True

    # stylize the image
    best_tensor = nsu.adam_optimization(content_tensor, [style_tensor],
                                        train_tensor,
                                        model,
                                        options,
                                        mask=mask,
                                        device=device,
                                        fileid='_' + str(i_s))

    # restart from the stylized image
    train_tensor = best_tensor.clone().detach()

# save the best image
nsu.imsave(best_tensor.cpu(), options.output)
