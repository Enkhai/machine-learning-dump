import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models import GMM


def normalized_squared_error(X, X_pred):
    return np.linalg.norm(X - X_pred) / X.shape[0]


if __name__ == '__main__':
    img = Image.open("data/im.jpg")
    X = np.array(img)

    plt.figure(figsize=(20, 16))
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X)
    plt.show()

    orig_shape = X.shape
    # flatten pixels
    X = X.reshape(-1, orig_shape[-1])
    # normalization
    X_mean, X_std = X.mean(0), X.std(0)
    X = (X - X.mean(0)) / X.std(0)

    n_comps = int(input("Input number of GMM components: "))

    # create model
    gmm = GMM(orig_shape[-1], n_comps)

    gmm.train(X, 100, 1e+3)
    X_pred = gmm.predict(X)

    # denormalization
    X_denorm = ((X * X_std) + X_mean).astype("uint8")
    X_pred_denorm = ((X_pred * X_std) + X_mean).astype("uint8")

    plt.figure(figsize=(20, 16))
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_pred_denorm.reshape(orig_shape))
    plt.show()

    plt.figure(figsize=(20, 16))
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_denorm.reshape(orig_shape))
    plt.show()

    segmentation_error = normalized_squared_error(X_denorm, X_pred_denorm)

    print("Segmentation error:", segmentation_error)
