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

    # resize F_XL into \hat F_XL
    features = input.view(a * b, c * d)

    # resize guidance channels
    guidance_channel_resized = F.interpolate(
        guidance_channel.unsqueeze(0).unsqueeze(0), size=(c, d))
    guidance_channel_view = guidance_channel_resized.view(1, c * d)
    guided_features = torch.mul(guidance_channel_view, features)

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


# recursive function to add conv layer to model
def build_model(cnn, normalization_mean, normalization_std, guidance_weights,
                guidance_channels, content_img, style_imgs, content_layers,
                style_layers, content_losses, style_losses):
    '''
    Subroutine that recoursively builds a model. It traverses the network
    until it finds a "Base" layers

    Arguments
    ----------
    'cnn' = input nn.Sequential model
    'normalization_mean' = RGB pixel mean
    'normalization_std' = RGB pixel standard deviation
    'guidance_channels' = masks to guide the image stylization
    'guidance_weights' = list containing the weights for the different
    'content_img' = content image
    'style_imgs' = list containing the style images
    'content_layers'= conv layers used for the content loss
    'style_layers'= conv layers used for the style losses
    'content_losses'= list containing the content losses
    'style_losses'= list containing the style losses
    '''

    # normalization module
    normalization = Normalization(normalization_mean,
                                  normalization_std).to(device)

    # initializa output model
    out_model = nn.Sequential(normalization)

    # go through the layers
    n_conv = 0
    for layer in cnn.modules():
        if isinstance(layer, nn.Conv2d):
            n_conv += 1
            name = 'conv_{}'.format(n_conv)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(n_conv)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(n_conv)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(n_conv)
        else:
            continue

        # add layer
        out_model.add_module(name, layer)

        # add content loss:
        if name in content_layers:
            target = out_model(content_img).detach()
            content_loss = ContentLoss(target)
            out_model.add_module("content_loss_{}".format(n_conv),
                                 content_loss)
            content_losses.append(content_loss)

        # add style loss:
        if name in style_layers:
            target_features = []
            for style_img in style_imgs:
                target_features.append(out_model(style_img).detach())
            style_loss = StyleLoss(target_features, guidance_channels,
                                   guidance_weights)
            out_model.add_module("style_loss_{}".format(n_conv), style_loss)
            style_losses.append(style_loss)

    return out_model, n_conv


# returns a new model, built from the old one, where normalization
# is in the first place and style and content losses are injected
# after the content and style layers.
def get_style_model_and_losses(cnn,
                               normalization_mean,
                               normalization_std,
                               guidance_channels,
                               guidance_weights,
                               content_img,
                               style_imgs,
                               content_layers=['conv_1'],
                               style_layers=['conv_1']):
    '''
    Returns a new model, built from the old one, where normalization
    is in the first place and style and content losses are injected
    after the content and style layers. It is meant for sequential
    models.

    Arguments
    ----------
    'cnn' = input nn.Sequential model
    'normalization_mean' = RGB pixel mean
    'normalization_std' = RGB pixel standard deviation
    'guidance_channels' = masks to guide the image stylization
    'guidance_weights' = list containing the weights for the different
                         stylizations
    'style_imgs' = list containing the style images
    'content_img' = content image
    'content_layers'= conv layers used for the content loss
    'style_layers'= conv layers used for the style losses
    '''

    # ------- recursively build the style transfer model -------

    # copy the base style transfer model
    cnn = copy.deepcopy(cnn)

    # list to be populated with losses
    content_losses = []
    style_losses = []

    # finally build model
    model, n_conv = build_model(cnn, normalization_mean, normalization_std,
                                guidance_weights, guidance_channels,
                                content_img, style_imgs, content_layers,
                                style_layers, content_losses, style_losses)

    # trim layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(
                model[i], StyleLoss):
            break

    model = model[:(n_conv + 1)]

    return model, style_losses, content_losses


# define the optimizer as in gatys
def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(cnn,
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

            # clears the gradient before each iteration
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
