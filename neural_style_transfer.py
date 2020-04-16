#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Pytorch
import torchvision.models as models
import neural_style_utils as nsu

# build command line argument parser
options = nsu.build_parser().parse_args()

# load the content and style images
content_img = nsu.imload(options.content, options.content_resize)
style_img = nsu.imload(options.style, options.style_resize)

# load pretrained model: whole model with classification head
model = models.squeezenet1_1(pretrained=True)

# stylize the image
best_img = nsu.adam_optimization(content_img, style_img, model, options)

# save the best image
nsu.imsave(best_img, options.output)