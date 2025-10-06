# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 01:42:58 2018

@author: Ceyx
"""
#part 1: importing

#coded methods
from split_selection import input_select
from data_encode import encode, split
from models import SVD
from datasets import CF_Dataset
from training import train_model, test_loss, set_model_untrained_weights

#libraries
import numpy as np #numpy
import pandas as pd #pandas
from pathlib import Path #path
import time #time
#import os #operating system
#import sys #system
import torch #PyTorch
#import torch.nn as nn #Neural network module
#import torch.nn.functional as F #functions
import matplotlib.pyplot as plt #plotting
import seaborn as sns

#sets the number of OpenMP threads to be used by PyTorch to the maximum.
#the default is half of that amount
#torch.set_num_threads(os.cpu_count())
#torch.set_num_threads(1)

#reproducibility
torch.manual_seed(0)
np.random.seed(0)

#part 2: preparing data

#import data
print(f'\nLoading the data...')
PATH = Path("../Datasets/ml-latest-small")
data = pd.read_csv(PATH/"ratings.csv")
    
#encode the dataset into unique and contiguous values
data = encode(data)
print(f'Dataset has been encoded')
    
num_ratings = len(data.rating)
num_users = len(data.userId.unique())
num_items = len(data.movieId.unique())
sparsity = num_ratings/(num_users*num_items)
mean_rating = data.rating.mean()
    
print(f'\nNumber of users: {num_users}')
print(f'Number of items: {num_items}')
print(f'Number of ratings: {num_ratings}')
print(f'Dataset sparsity: {sparsity}')
print(f'\nAverage rating: {mean_rating}\n')
            
#splitting variables input
method, value = input_select()
    
print(f'\nSplitting:\nMethod: {method}\nValue: {value}')
    
#dataset splitting
print('\nSplitting data into training and testing set\n')    
train_set, test_set = split(data, method, value)
    
print(f'\nTraining set: {train_set}\n')
print(f'\nTesting set: {test_set}\n')
    
print(f'\nMaking the torch datasets\n')
train_dataset = CF_Dataset(train_set)
test_dataset = CF_Dataset(test_set)

#epochs_array = [5, 10, 20, 30, 40, 50, 60, 70]
epochs_array = [100]
#decay_array = [0.0007, 0.0005, 0.0004]

#make a seaborn set
sns.set()

pal = sns.color_palette("Set2", 3)

fig = plt.figure(figsize=(8,8)) #8x8

#for i, decay in enumerate(decay_array):

test_mse = []
    
for epochs in epochs_array:

    #part 3: modeling
        
    model = SVD(num_users, num_items, mean_rating, emb_size=100)
    #set the model to cuda for gpu training
    model = model.cuda()
    print(f'Model initialized: \n{model}')
        
    #the train function automatically sets the model to cuda
    #by reinitializing it as a torch.nn.DataParallel module
    #when the cuda parameter is set to True.
    #cuda = False will initialize a training session based
    #solely on cpu computations.
    #in the case of a cuda based training, even if the model
    #had initially been set to cpu, in order to calculate the
    #the untrained weights and get the test loss, cuda must
    #also be set to True
        
    #part 4: training
    
    start = time.time()
    train_model(model, train_dataset, lr=0.01, wd=0.00028, epochs=epochs, cuda=True)
    end = time.time()
    print(f'\nTime elapsed for training: {end-start} sec')
        
    #average model weights for untrained users/items
    set_model_untrained_weights(model, train_dataset, test_dataset, cuda=True)
        
    #part 5: testing
        
    test_mse += [test_loss(model, test_dataset, cuda=True)]
    
subplot = fig.add_subplot(1,1,1)    
#draw the following
#alpha (intensity) = 0.5
subplot.plot(epochs_array, test_mse, c=pal[0], label='Test loss', alpha=0.5, linewidth=5)

#draw the legend at the best location, fonts: 20px
plt.legend(loc='best', fontsize=20)
#make the ticks as following
plt.xticks(fontsize=16);
plt.yticks(fontsize=16);
#and the labels
plt.xlabel('Epochs', fontsize=30);
plt.ylabel('MSE', fontsize=30);
plt.show()