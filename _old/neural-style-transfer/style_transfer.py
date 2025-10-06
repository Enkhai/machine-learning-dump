import matplotlib.pyplot as plt

import torch
from torchvision import models

from helper import load_image
from train_style import train_image_style

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg = models.vgg19(pretrained=True).features.to(device)
# torch.save(vgg, 'model.pht')
# vgg = torch.load('model.pth').to(device)

# load the content and style images
content = load_image('images/house.jpg').to(device)
style = load_image('images/cubism.jpg', shape=content.shape[-2:]).to(device)

for image in train_image_style(vgg, content, style):
    plt.imshow(image)
    plt.show()
