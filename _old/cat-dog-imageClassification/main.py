import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, SubsetRandomSampler
from collections import OrderedDict
import time
import numpy as np
from random import shuffle

from training import train, test

# device agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = 'Cat_Dog_data'

if __name__ == '__main__':
    # make the training and testing transformations
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5],
                                                                [0.5, 0.5, 0.5])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])

    # load the data from our folder dataset
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    # let's create validation samples for our training
    test_data_indices = list(range(len(test_data)))
    shuffle(test_data_indices)
    split = int(np.floor(len(test_data_indices) * 0.8))
    test_indices, validation_indices = test_data_indices[split:], test_data_indices[:split]

    test_sampler = SubsetRandomSampler(test_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)

    # and make our data loaders
    trainloader = DataLoader(train_data, batch_size=320, shuffle=True)
    testloader = DataLoader(test_data, batch_size=320, sampler=test_sampler)
    validationloader = DataLoader(test_data, batch_size=320, sampler=validation_sampler)

    # we will use a pre-trained network
    # pre-trained networks can function well as feature detectors
    model = models.densenet201(pretrained=True)

    # freeze the model parameters
    for param in model.parameters():
        param.requires_grad = False

    # and replace the last classifier layer with our own
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(1920, 800)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(800, 2)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier

    # set our model to the preferred device
    model = model.to(device)

    # since we use LogSoftmax as the output, we will use negative likelihood loss as our error function
    criterion = nn.NLLLoss()
    # just Adam
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # calculate the time taken for training
    start = time.time()
    train(model, criterion, optimizer, trainloader, validationloader, epochs=5, device=device)
    end = time.time()
    print("Time elapsed for training: %f seconds" % (end - start))

    # and calculate our error loss
    test(model, criterion, testloader, device=device)
