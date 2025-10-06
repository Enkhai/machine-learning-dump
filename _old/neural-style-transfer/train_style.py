from torch import optim
import torch.nn.functional as F

from helper import get_features, gram_matrix, im_convert


def train_image_style(model, content, style, target=None, steps=2000, alpha=1, beta=1e6, show_every=400,
                      style_weights=None):

    # freeze model weights
    for param in model.parameters():
        param.requires_grad_(False)

    # get the content and style features from the model
    content_features = get_features(content, model)
    style_features = get_features(style, model)

    # if a target is not specified use a clone of the content as a target
    if target is None:
        target = content.clone()

    # we need to update the target image
    target = target.requires_grad_(True)

    # if style weights have not been initialized set them
    if style_weights is None:
        style_weights = {'conv1_1': 1.,
                         'conv2_1': 0.8,
                         'conv3_1': 0.5,
                         'conv4_1': 0.3,
                         'conv5_1': 0.1}

    # we need to update the target image
    optimizer = optim.Adam([target], lr=0.003)
    # we will need the gram matrices of the style features for later
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    for ii in range(1, steps + 1):
        # get the target features
        target_features = get_features(target, model)
        # step 1: calculate the content loss
        content_loss = F.mse_loss(target_features['conv4_2'], content_features['conv4_2'])

        style_loss = 0

        # for each style layer calculate the style loss
        for layer in style_weights:
            target_feature = target_features[layer]
            _, d, h, w = target_feature.shape

            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            # step 2: calculate the style loss
            layer_style_loss = style_weights[layer] * F.mse_loss(target_gram, style_gram)

            # add the layer loss to the total style loss
            style_loss += layer_style_loss / (d * w * h)

        # step 3: calculate the total loss
        total_loss = alpha * content_loss + beta * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if ii % show_every == 0:
            print("Total loss: ", total_loss.item())
            # yield a PIL image for every iteration
            yield im_convert(target)
