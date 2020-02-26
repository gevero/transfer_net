# importing libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

# define devic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# content loss
class ContentLoss(nn.Module):
    def __init__(
        self,
        target,
    ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


# gram matrix definition
def gram_matrix(input, guidance_channel, guidance_weight):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    # resize guidance channels
    guidance_channel_resized = F.interpolate(
        guidance_channel.unsqueeze(0).unsqueeze(0), size=(c, d))
    guidance_channel_view = guidance_channel_resized.view(1, c * d)
    guided_features = torch.mul(guidance_channel_view, features)

    # resize F_XL into \hat F_XL
    features = input.view(a * b, c * d)

    # compute the gram product
    G = torch.mm(guided_features, guided_features.t())

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return guidance_weight * G.div(a * b * c * d)


# style loss function
class StyleLoss(nn.Module):
    def __init__(self, target_features, guidance_channels, guidance_weights):
        super(StyleLoss, self).__init__()

        # initialize channels and weights
        self.target = []
        self.guidance_channels = guidance_channels.detach()
        self.guidance_weights = guidance_weights.detach()

        # build guided targets
        for target_feature, guidance_channel, guidance_weight in zip(
                target_features, guidance_channels, guidance_weights):
            self.target.append(
                gram_matrix(target_feature, guidance_channel,
                            guidance_weight).detach())

    def forward(self, input):

        # compute loss for all guided gram matrixes
        self.loss = 0
        for target, guidance_channel, guidance_weight in zip(
                self.target, self.guidance_channels, self.guidance_weights):
            G = gram_matrix(input, guidance_channel, guidance_weight)
            self.loss += F.mse_loss(G, target)

        return input


# define normalization class to plug into the model
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# returns a new model, built from the old one, where normalization
# is in the first place and style and content losses are injected
# after the content and style layers
def get_style_model_and_losses(cnn,
                               normalization_mean,
                               normalization_std,
                               guidance_channels,
                               guidance_weights,
                               style_imgs,
                               content_img,
                               content_layers=['conv_1'],
                               style_layers=['conv_1']):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean,
                                  normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(
                layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_features = []
            for style_img in style_imgs:
                target_features.append(model(style_img).detach())
            style_loss = StyleLoss(target_features, guidance_channels,
                                   guidance_weights)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(
                model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


# define the optimizer as in gatys
def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(
    cnn,
    # style transfer main function
    normalization_mean,
    normalization_std,
    guidance_channels,
    guidance_weights,
    content_img,
    style_imgs,
    input_img,
    num_steps=300,
    style_weight=1000000,
    content_weight=1,
    content_layers=['conv_1'],
    style_layers=['conv_1']):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn,
        normalization_mean,
        normalization_std,
        guidance_channels,
        guidance_weights,
        style_imgs,
        content_img,
        content_layers=content_layers,
        style_layers=style_layers)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img