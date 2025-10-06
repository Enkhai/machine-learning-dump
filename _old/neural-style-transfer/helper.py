from PIL import Image
from io import BytesIO
import requests
import numpy as np

import torch
from torchvision import transforms


# returns a tensor from an image
def load_image(img_path, image=None, max_size=512, shape=None):
    # if an image is not given the img_path should matter
    if image is None:
        # http url case
        if "http" in img_path:
            response = requests.get(img_path)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(img_path).convert('RGB')

    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    # define transforms
    # vgg networks are trained on images with each channel normalized by
    # - mean=[0.485, 0.456, 0.406]
    # - std=[0.229, 0.224, 0.225]
    in_transform = transforms.Compose([transforms.Resize(size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),
                                                            (0.229, 0.224, 0.225))])
    # discard the transparent (alpha) channel (:3) and add the batch dimension
    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image


# returns a PIL image from a tensor
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()  # discard batch dimension
    image = image.permute(1, 2, 0)  # permute image dimensions, color channels will go last
    # de-normalize with the vgg network mean and std
    image = image * torch.tensor((0.229, 0.224, 0.225)) + torch.tensor((0.485, 0.456, 0.406))
    image = image.clamp(0, 1)

    # return an image from numpy array
    return Image.fromarray((image * 255).numpy().astype(np.uint8))


# run an image forward through a vgg19 model and get the features for a set of layers
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1'}

    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def gram_matrix(tensor):
    b, d, h, w = tensor.size()
    tensor = tensor.view(b * d, h * w)
    return tensor.mm(tensor.t())
