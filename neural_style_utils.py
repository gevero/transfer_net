#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Miscellaneous libraries
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import numpy as np
from collections import namedtuple
from functools import singledispatch

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import neural_style_params as nsp

# function to convert to pil images
toImage = transforms.ToPILImage()


def build_parser():
    '''
    A parser builder to handle the command line arguments
    '''

    desc = 'a light pytorch implementation of neural style using squeezeNet.'
    parser = ArgumentParser(description=desc)
    parser.add_argument('--content',
                        dest='content',
                        help='path to content image',
                        required=True)
    parser.add_argument('--style',
                        dest='style',
                        action='append',
                        help='path to style image',
                        required=True)
    parser.add_argument('--output',
                        dest='output',
                        help='output path e.g. output.jpg',
                        default=nsp.OUTPUT_PATH)
    parser.add_argument('--style-weight',
                        type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        default=nsp.STYLE_WEIGHT)
    parser.add_argument('--content-weight',
                        type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        default=nsp.CONTENT_WEIGHT)
    parser.add_argument(
        '--content-resize',
        type=int,
        dest='content_resize',
        help='resize so that size of smaller edge match the given size',
        default=None)
    parser.add_argument(
        '--style-resize',
        type=int,
        dest='style_resize',
        help='resize so the size of smaller edge match the given size',
        default=nsp.STYLE_RESIZE)
    parser.add_argument(
        '--size-steps',
        type=int,
        dest='size_steps',
        help='iterates stylization for size_steps to get to final size',
        default=nsp.SIZE_STEPS)
    parser.add_argument('--iterations',
                        type=int,
                        dest='iters',
                        help='iterations (default %(default)s)',
                        default=nsp.ITERATIONS)
    parser.add_argument('--learning-rate',
                        type=float,
                        dest='lr',
                        help='learning-rate for adam',
                        default=nsp.LEARNING_RATE)
    parser.add_argument(
        '--report-interval',
        type=int,
        dest='report_intvl',
        help='report loss and current image every interval number',
        default=nsp.REPORT_INTERVAL)
    parser.add_argument('--imsave-interval',
                        type=int,
                        dest='imsave_intvl',
                        help='save image every interval number',
                        default=nsp.IMSAVE_INTERVAL)
    # group for content or random start
    start_image_group = parser.add_mutually_exclusive_group(required=False)
    start_image_group.add_argument(
        '--random-start',
        dest='random_start',
        help='start stylized image from random tensor',
        action='store_true')
    start_image_group.add_argument(
        '--content-start',
        dest='random_start',
        help='start stylized image from content img',
        action='store_false')
    parser.set_defaults(feature=nsp.RANDOM_START)
    # group for standard training
    std_train_group = parser.add_mutually_exclusive_group(required=False)
    std_train_group.add_argument(
        '--standard-train',
        dest='std_tr',
        help='standard train: lr=0.1 for 500 step and then lr=0.01 for 500 \
            step',
        action='store_true')
    start_image_group.add_argument(
        '--custom-train',
        dest='std_tr',
        help='use custom iterations and leaarning rates',
        action='store_false')
    parser.set_defaults(feature=nsp.STANDARD_TRAIN)

    return parser


def imload(image_name, resize=None):
    '''
    A function to load an image and convert it to a Pytorch tensor

    Arguments
    ---------
    'image_name' = a string. The image name.
    'resize' = an interger or a tuple (integer,integer). Image
               new size.

    Returns
    -------
    'image' = a (1 x C x H x W) tensor.
    '''

    # load image
    image = Image.open(image_name)

    # resize if necessary
    if resize is not None:
        resizefunc = transforms.Resize(resize)
        image = resizefunc(image)

    # Build transform (H x W x C), [0, 255] --> (C x H x W), [0.0, 1.0]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # PIL Image to (1 x C x H x W) Tensor
    with torch.no_grad():
        image = transform(image)
        image = image.unsqueeze(0)

    return image


def tensor_resize(tensor, size):
    '''
    A function to load an image and convert it to a Pytorch tensor

    Arguments
    ---------
    'tensor' = a (1 x C x H x W) tensor.
    'size' = (h,w) resizing tuple.

    Returns
    -------
    'image' = a (1 x C x H x W) rescaled tensor.
    '''

    # auxiliary function
    resizefunc = transforms.Resize(size)
    toImage = transforms.ToPILImage()
    toTensor = transforms.ToTensor()

    # resize
    with torch.no_grad():
        tensor_img = toImage(tensor.squeeze().cpu())
        tensor_resized = toTensor(resizefunc(tensor_img)).unsqueeze(0)

    return tensor_resized


def imshow(img):
    '''
    Convert torch tensor to PIL image and then show image inline.

    Arguments
    ---------
    'img' = a PIL image.
    '''

    # denormalize tensor
    img = toImage(img[0] * 0.5 + 0.5)
    plt.imshow(img, aspect=None)
    plt.axis('off')
    plt.gcf().set_size_inches(8, 8)
    plt.show()


def imsave(img, path):
    '''
    Convert torch tensor to PIL image and then save to path.

    Arguments
    ---------
    'img' = a PIL image.
    '''

    # denormalize tensor before convert
    img = toImage(img[0] * 0.5 + 0.5)
    img.save(path)


def size_array(content_size, size_steps):
    '''
    A function that computes an array of tuples representing
    all the different size steps to get to the final stylization.

    Arguments
    ---------
    'content_size' = a tuple with the stylized image final size.
    'size_step' = an integer. The steps to get to the final size.

    Returns
    -------
    'size_array' = an array of tuples containing all the size steps
    '''

    # if only one size step -> original size
    if size_steps == 1:
        return [content_size]

    # if multiple size steps -> size array
    h = content_size[0]
    w = content_size[1]
    min_size = nsp.CONTENT_START_SIZE
    max_size = min(content_size)
    s_array = []
    for s in np.linspace(min_size, max_size, size_steps):

        s = int(s)

        if h > w:
            s_array.append((s * (h // w), s))
        else:
            s_array.append((s, s * (h // w)))

    return s_array


class FeatureExtractor(nn.Module):
    '''
    A nn.Module class to extract a intermediate activation of a Torch module
    '''
    def __init__(self, module):
        '''
        Initializes the feature extractor.

        Arguments
        ---------
        'module' = a pytorch model which computes the features
                   necessary to the neural style transfer process
        '''

        super().__init__()
        self.module = module

    def forward(self, image, layers, mask=None):
        '''
        Forward pass.

        Arguments
        ---------
        'image' = input image tensor.
        'layers' = list containing the layer positions used
            to compute the needed features.
        'mask' = a one hot encoded mask tensor to perform
            guided style transfer.
        '''

        # compute features
        features = []
        for i in range(layers[-1] + 1):
            image = self.module[i](image)
            if i in layers:
                features.append(image)

        # create guidance channels
        if mask is not None:

            # resize one hot masks to get channels
            feature_channels = []
            for feature in features:

                # get feature size
                n, f, h, w = feature.size()

                # interpolate and store
                channel = F.interpolate(mask.unsqueeze(0).unsqueeze(0),
                                        size=(h, w))
                feature_channels.append(channel)

            return features, feature_channels

        return features


class GramMatrix(nn.Module):
    '''
    A nn.Module class to build gram matrix from style features.
    The style features are the output of FeatureExtractor
    forward pass.
    '''
    def forward(self, style_features, feature_channels=None):
        '''
        Forward pass.

        Arguments
        ---------
        'style_features' = input feature tensors from FeatureExtractor.
        'feature_channels' = style channel tensors from FeatureExtractor.
        '''

        # empty output list
        gram_features = []

        # guided styles
        if feature_channels is not None:

            for feature, channel in zip(style_features, feature_channels):

                # get feature size
                n, f, h, w = feature.size()

                # reshape channel
                channel = channel.resize(1, h * w)

                # guide style features and compute gram matrix
                feature = feature.resize(n * f, h * w)
                feature = torch.mul(channel, feature)
                gram_features.append(
                    (feature @ feature.t()).div_(2 * n * f * h * w))

        # single style
        else:
            for feature in style_features:
                n, f, h, w = feature.size()
                feature = feature.resize(n * f, h * w)
                gram_features.append(
                    (feature @ feature.t()).div_(2 * n * f * h * w))

        return gram_features


class Stylize(nn.Module):
    '''
    A nn.Module class to compute style and content features.
    - feature is an instance of FeatureExtractor
    - gram is an instance of GramMatrix
    '''
    def __init__(self, feature, gram):
        '''
        Initializes the feature extractor.

        Arguments
        ---------
        'feature' = a FeatureExtractor instance.
        'gram' = a GramMatrix instance.
        '''
        super().__init__()

        # initialize layer with FeatureExtractor and GramMatrix
        self.feature = feature
        self.gram = gram

    def forward(self, x, feature_channels=None):
        '''
        Forward pass.

        Arguments
        ---------
        'x' = tensor the shall become the stylized image.
        '''

        # style features
        s_feats = self.feature(x, nsp.STYLE_LAYER)
        s_feats = self.gram(s_feats, feature_channels)

        # content features
        c_feats = self.feature(x, nsp.CONTENT_LAYER)

        return s_feats, c_feats


def totalloss_singlestyle(custom_loss, style_refs, content_refs,
                          style_features, content_features, style_weight,
                          content_weight):
    '''
    A function that computes the total loss.

    Arguments
    ---------
    'custom_loss' = pytorch loss function.
    'style_refs, content_refs' = tensors. Style and content features
        for the reference images.
    'style_features, content_features' = tensors. Style and content features
        for the final image.
    'style_weight, content_weight)' = list. Weights for the computation of the
        style and content loss

    Returns
    -------
    'total_loss' = a 0-dim tensor with the loss?
    '''

    # style loss: the contributions of all layers are normalized
    style_loss = [
        custom_loss(style_features[i], style_refs[i])
        for i in range(len(style_features))
    ]
    mean_loss = sum(style_loss).item() / len(style_features)
    style_loss = sum([(mean_loss / l.item()) * l * nsp.STYLE_LAYER_WEIGHTS[i]
                      for i, l in enumerate(style_loss)]) / len(style_features)

    # content loss
    content_loss = sum([
        custom_loss(content_features[i], content_refs[i])
        for i in range(len(content_refs))
    ]) / len(content_refs)

    # total loss, weighting style and content
    # print(style_weight, content_weight)
    # print('styleloss: ', style_weight * style_loss, ' contentloss: ',
    #       content_weight * content_loss)
    total_loss = style_weight * style_loss + content_weight * content_loss

    return total_loss


def totalloss(custom_loss,
              train_img,
              style_refs,
              content_refs,
              style,
              options,
              style_channels=None):
    '''
    A function that computes the total loss.

    Arguments
    ---------
    'custom_loss' = pytorch loss function.
    'train_img' = tensor the shall become the stylized image.
    'style_refs' = tensors or list of. Style features of the reference images.
    'content_refs' = tensors. Content features for the reference images.
    'style_features, content_features' = tensors. Style and content features
        for the final image.
    'style' = Stylize class instance.
    'options' = parser options.
    'style_channels' = list of tensors. Guidance chanels for the transfer
        of multiple styles.

    Returns
    -------
    'total_loss' = a 0-dim tensor with the loss?
    '''

    # multiple styles
    if style_channels is not None:

        # compute composite loss for multiple styles
        loss = 0
        for style_ref, style_channel in zip(style_refs, style_channels):

            # compute train_img style and content features
            style_features, content_features = style(train_img, style_channel)

            # the total loss
            loss += totalloss_singlestyle(custom_loss, style_ref, content_refs,
                                          style_features, content_features,
                                          options.style_weight,
                                          options.content_weight)

    # single style
    else:

        # compute train_img style and content features
        style_features, content_features = style(train_img)

        # the total loss
        loss = totalloss_singlestyle(custom_loss, style_refs, content_refs,
                                     style_features, content_features,
                                     options.style_weight,
                                     options.content_weight)
    return loss


def reference_singlestyle(style_img, content_img, feature, gram, mask=None):
    '''
    A function that computes the reference style and content features.

    Arguments
    ---------
    'style_img' = a tensor. The style reference image.
    'content_img' = a tensor. The content reference image.
    'feature' = a FeatureExtractor instance.
    'gram' = a GramMatrix instance.
    'mask' = a one hot encoded mask tensor to perform
            guided style transfer.

    Returns
    -------
    'style_refs, content_refs, (style_channels)' = tensors.
        The style and content reference features to
        calculate the loss, + style guidance channels.
    '''

    # style reference features
    if mask is not None:
        style_refs, style_channels = feature(style_img, nsp.STYLE_LAYER, mask)
        style_refs = gram(style_refs, style_channels)
        style_channels = [
            style_channel.detach() for style_channel in style_channels
        ]
    else:
        style_refs = feature(style_img, nsp.STYLE_LAYER)
        style_refs = gram(style_refs)

    # unrolling in a list of tensors
    style_refs = [style_ref.detach() for style_ref in style_refs]

    # content reference features: expand the tensor in a list of tensors?!?
    content_refs = feature(content_img, nsp.CONTENT_LAYER)
    # unrolling in a list of tensors
    content_refs = [content_ref.detach() for content_ref in content_refs]

    if mask is not None:
        return style_refs, content_refs, style_channels
    else:
        return style_refs, content_refs


def reference(style_img, content_img, feature, gram, mask=None):
    '''
    A function that computes the reference style and content features.

    Arguments
    ---------
    'style_img' = a tensor or a list of tensors. The style reference images.
    'content_img' = a tensor. The content reference image.
    'feature' = a FeatureExtractor instance.
    'gram' = a GramMatrix instance.
    'mask' = a categorically encoded mask tensor to perform
            guided style transfer.

    Returns
    -------
    'style_refs, content_refs' = tensors.
        The style and content reference features to
        calculate the loss.

    or

    style_refs_list, content_refs, style_channels_list' = tensors or list of.
        The list contains style_refs and style_channels for each style, while
        the content refs are still unique.
    '''

    # multiple styles
    if mask is not None:

        # mask to one-hot encoding
        moh = F.one_hot(mask.long()).float()

        style_refs_list = []
        style_channels_list = []
        for i_img, s_img in enumerate(style_img):

            # compute reference feature and style channels
            style_refs, content_refs, style_channels = reference_singlestyle(
                s_img, content_img, feature, gram, moh[:, :, i_img])

            # save data
            style_refs_list.append(style_refs)
            style_channels_list.append(style_channels)

        return style_refs_list, content_refs, style_channels_list

    # single style
    else:
        style_refs, content_refs = reference_singlestyle(
            style_img[0], content_img, feature, gram)
        return style_refs, content_refs


def adam_optimization(content_img,
                      style_img,
                      train_img,
                      model,
                      options,
                      mask=None,
                      device='cpu',
                      fileid=''):
    '''
    A function that computes the reference style and content features.

    Arguments
    ---------
    'content_img' = a tensor. The content reference image.
    'style_img' = a tensor or a list of tensors. The style reference images.
    'train_img' = a tensor. The stylized image.
    'model' = a PyTorch model.
    'options' = parser options.
    'mask' = a categorically encoded mask tensor to perform
            guided style transfer.


    Returns
    -------
    'best_img' = a tensor. Best image found.
    '''

    # ----------------------------------------------------------------
    # --------- Initialize loss, tensors, optim and trackers ---------
    # ----------------------------------------------------------------

    # loss function
    l2loss = nn.MSELoss(size_average=False)

    # optimizer: different optimizers for different phases of the training
    optimizer = optim.Adam([train_img], lr=options.lr)
    optimizer1 = optim.Adam([train_img], lr=0.1)
    optimizer2 = optim.Adam([train_img], lr=0.01)

    # trackers
    loss_history = []
    min_loss = float('inf')
    best_img = None

    # --------------------------------------------
    # --------- Initialize style classes ---------
    # --------------------------------------------

    # get headless model
    headless_model = next(model.children())
    headless_model = nn.Sequential(
        *list(headless_model.children())[:-1]).to(device)
    print(headless_model)

    # instantiate classes to compute style and content features
    feature = FeatureExtractor(headless_model)
    gram = GramMatrix()
    style = Stylize(feature, gram)

    # ---------------------------------------------------------------
    # --------- Initialize reference style/content features ---------
    # ---------------------------------------------------------------

    # multiple styles
    if mask is not None:
        # compute reference feature and style channels
        style_refs_list, content_refs, style_channels_list = reference(
            style_img, content_img, feature, gram, mask)

    # single style
    else:
        style_refs, content_refs = reference(style_img, content_img, feature,
                                             gram)

    # ---------------------------------------------------------------
    # ----------------------- optimization loop ---------------------
    # ---------------------------------------------------------------

    Start = datetime.now()
    num_iters = 800 if options.std_tr else options.iters

    for i in range(num_iters):

        # clears the gradient before each iteration
        # with pytorch you need to clear them
        # explicitly
        if options.std_tr:
            if i <= num_iters // 2:
                optimizer1.zero_grad()
            else:
                optimizer2.zero_grad()
        else:
            optimizer.zero_grad()

        # clamp the values of the input image
        train_img.data.clamp_(-1, 1)

        # ---------------------------------------------------------------
        # --------- Style features and loss: maybe for multi-styles -----
        # ---------------------------------------------------------------
        # multiple styles
        if mask is not None:
            loss = totalloss(l2loss, train_img, style_refs_list, content_refs,
                             style, options, style_channels_list)

        # single style
        else:
            loss = totalloss(l2loss, train_img, style_refs, content_refs,
                             style, options)

        # saving the loss history
        loss_history.append(loss.item())

        # save best image result before image update
        if min_loss > loss_history[-1]:
            min_loss = loss_history[-1]
            best_img = train_img

        # BACKWARD PASS
        loss.backward()

        if options.std_tr:
            if i < num_iters // 2:
                optimizer1.step()
            else:
                optimizer2.step()
        else:
            optimizer.step()

        # report loss and image
        if i % options.report_intvl == 0:
            print("step: %d loss: %f,time per step:%s s" %
                  (i, loss_history[-1],
                   (datetime.now() - Start) / options.report_intvl))
            Start = datetime.now()

        # save image every imsave_intvl
        if i % options.imsave_intvl == 0 and i != 0:
            save_path = options.output.replace('.jpg',
                                               fileid + '_step%d.jpg' % i)
            imsave(train_img.cpu(), save_path)
            print("image at step %d saved." % i)

    return best_img


class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]
