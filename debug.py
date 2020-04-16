# ------------ import libraires ------------
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image

# transfer net
import transfer_utils as tu

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------ parameters ------------
# models = [VGG19,SqueezeNet]
model = 'SqueezeNet'

# desired depth layers to compute style/content losses
content_layers_default = ['conv_2', 'conv_3', 'conv_4']
style_layers_default = ['conv_2', 'conv_3', 'conv_4']

# list of style images
content_name = './content/portrait.jpg'
style_names = ['./style/hr-color1.jpg']
n_styles = len(style_names)

# desired size of the output image
h_orig = 1920
w_orig = 1949
h_im = int(1920)
w_im = int(1949)
h_large = int(1920)
w_large = int(1949)


# ------------ aux functions ------------
def content_loader(image_name, loader, h_img, w_img):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def style_loader(style_names, loader, h_img, w_img):

    # create style_imgs tensor
    n_styles = len(style_names)
    m_styles = torch.ones((n_styles, 1, 3, h_img, w_img))

    # store all the images
    for i_sn, style_name in enumerate(style_names):
        image = Image.open(style_name)
        image = loader(image).unsqueeze(0)
        m_styles[i_sn] = image

    return m_styles.to(device, torch.float)


# tensor loaders
loader = transforms.Compose(
    [transforms.Resize((h_im, w_im)),
     transforms.ToTensor()])

loader_hires = transforms.Compose(
    [transforms.Resize((h_large, w_large)),
     transforms.ToTensor()])

# ------------ start real work ------------
# create model
if model == 'VGG19':
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
else:
    cnn = models.squeezenet1_0(pretrained=True).features.to(device).eval()
cnn_norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

print(cnn)

style_imgs = style_loader(style_names, loader, h_im, w_im)
content_img = content_loader(content_name, loader, h_im, w_im)

assert style_imgs[0].size() == content_img.size(), \
    "we need to import style and content images of the same size"

# create guidance channel
guidance_weigths = torch.ones(n_styles, device=device)
guidance_weigths = guidance_weigths / guidance_weigths.sum()
guidance_channels = torch.ones((n_styles, h_im, w_im), device=device)

# create input image
input_img = content_img.clone()

# create the model
model, style_losses, content_losses = tu.get_style_model_and_losses(
    cnn,
    cnn_norm_mean,
    cnn_norm_std,
    guidance_channels,
    guidance_weigths,
    content_img,
    style_imgs,
    content_layers_default,
    style_layers=['conv_2', 'conv_5', 'conv_6'])

print(model)
