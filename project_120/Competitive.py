import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns


class Competitive(object):
    def __init__(self, input_dim, num_neurons, lr, max_epochs, low, high):
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.W = np.random.uniform(low, high, (num_neurons, input_dim))
        self.lr = lr
        self.max_epochs = max_epochs
        self.iter = 0

    def train(self, X, num_epochs):
        for epoch in range(num_epochs):
            self.iter += 1
            for x in X:
                min_idx = self.closest_idx(x)
                self.W[min_idx] += self.lr * (x - self.W[min_idx])
            self.lr *= 1 - self.iter / self.max_epochs

    def predict(self, X):
        return self.closest_idx(X)

    def closest_idx(self, x):
        return np.array([np.linalg.norm(x - self.W[i], axis=-1) for i in range(self.num_neurons)]).argmin(0)


def plot_training(model, X_train, t_train):
    fig, axes = plt.subplots(2, 1)
    for ax in axes:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    n_iter = 100

    # Plot 1 - static
    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 2],
                    hue=t_train, style=t_train,
                    ax=axes[0])

    for i in range(n_iter):
        axes[1].clear()

        axes[0].set_title('Epoch ' + str(i + 1) + ' out of ' + str(n_iter))
        model.train(X_train, 1)

        train_pred = model.predict(X_train)

        neurons = model.W
        axes[1].scatter(neurons[:, 0], neurons[:, 2], c=range(len(neurons)))
        sns.scatterplot(x=X_train[:, 0], y=X_train[:, 2],
                        hue=train_pred, style=train_pred,
                        ax=axes[1])

        fig.canvas.draw()
        plt.pause(0.1)


if __name__ == '__main__':
    data = load_iris()

    data_low, data_high = data.data.min(0), data.data.max(0)

    X_train, X_test, t_train, t_test = train_test_split(data.data, data.target, stratify=data.target)
    X_train = X_train[t_train.argsort()]
    t_train.sort()
    X_test = X_test[t_test.argsort()]
    t_test.sort()

    model = Competitive(4, 4, 1, 300, data_low, data_high)
    model.train(X_train, 100)

    pred = model.predict(X_train)
    plot_training(model, X_train, t_train)
