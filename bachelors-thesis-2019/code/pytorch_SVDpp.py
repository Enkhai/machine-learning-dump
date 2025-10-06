# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 06:59:45 2018

@author: Ceyx
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 01:42:58 2018

@author: Ceyx
"""
#part 1: importing

#coded methods
from data_split import split
from split_selection import input_select
from data_encode import encode, rating_idx
from models import SVD_pp
from training import train_model, test_loss

#libraries
#import numpy as np #numpy
import pandas as pd #pandas
from pathlib import Path #path
import time #time
#import torch #PyTorch
#import torch.nn as nn #Neural network module
#import torch.nn.functional as F #functions

#part 2: preparing data

#import data
PATH = Path("J:\Shared folder\TEI\It has begun\Datasets\ml-latest-small")
data = pd.read_csv(PATH/"ratings.csv")

#number of users, items, ratings, maximum index of items,
#average rating and data sparsity
raw_num_ratings = len(data.rating)
raw_num_users = len(data.userId.unique())
raw_num_items = len(data.movieId.unique())
max_item_idx = data.movieId.max()
mean_rating = data.rating.mean()
sparsity = raw_num_ratings/(raw_num_users*raw_num_items)

print(f'\nNumber of users: {raw_num_users}')
print(f'\nNumber of items: {raw_num_items}')
print(f'\nLargest item index: {max_item_idx}')
print(f'\nNumber of ratings: {raw_num_ratings}')
print(f'\nAverage rating: {mean_rating}\n')
print(f'\nDataset sparsity: {sparsity}')

#*IMPLICIT FEEDBACK MODELS ONLY*
#return an array of rating positions (user-to-item) within
#the rating matrix
ratingidx = rating_idx(data.userId.values, data.movieId.values)

#splitting variables input
method, value = input_select()

print(f'\nMethod: {method}\nValue: {value}')

#dataset splitting
train_set, test_set = split(data, method, value)

#encoding dataset
encode(train_set)
encode(test_set)

print(f'\nTraining set: {train_set}')
print(f'\nTesting set: {test_set}')

#part 3: modeling
'''
train_num_users = len(train_set.userId.unique())
train_num_items = len(train_set.movieId.unique())
'''

model = SVD_pp(raw_num_users, max_item_idx+1, ratingidx, mean_rating)

#part 4: training
train_users = train_set.userId.values
train_items = train_set.movieId.values
train_ratings = train_set.rating.values

start = time.time()
train_model(model, train_users, train_items, train_ratings)
end = time.time()
print(f'\nTime elapsed for training: {end-start} sec')

#part 5: testing

test_users = test_set.userId.values
test_items = test_set.movieId.values
test_ratings = test_set.rating.values

test_loss(model, test_users, test_items, test_ratings)