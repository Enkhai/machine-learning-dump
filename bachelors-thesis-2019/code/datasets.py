#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 19:51:42 2019

@author: ceyx
"""

import torch
from torch.utils.data import Dataset

#collaborative filtering dataset based on the MovieLens datasets
class CF_Dataset(Dataset):
    
    #data initiliaziation is best done in the init method.
    #optimize code to reduce training time
    def __init__(self, data):
        self.data = data
        self.length = len(data)
        self.users = torch.LongTensor(data['userId'].values)
        self.items = torch.LongTensor(data['movieId'].values)
        self.ratings = torch.FloatTensor(data['rating'].values)
        self.timestamp = torch.LongTensor(data['timestamp'].values)
    
    def __getitem__(self, idx):
        return {'userId':self.users[idx], 'movieId':self.items[idx], 'rating':self.ratings[idx], 'timestamp':self.timestamp[idx]}
    
    def __len__(self):
        return self.length
    