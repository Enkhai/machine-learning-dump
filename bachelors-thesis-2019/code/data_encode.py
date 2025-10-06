# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 04:33:04 2018

@author: Ceyx
"""
import numpy as np
import sys
from sklearn.utils import shuffle

#encodes columns of dataframe data into unique values
#and drops all negative records.
#the dataframe data is returned
def encode(data, columns=['userId', 'movieId']):
    for col_name in columns:
        _,col,_ = proc_col(data[col_name])
        data[col_name] = col
        data = data[data[col_name] >= 0]
    return data

#processes a column and returns a dictionary of its indices
#and values, an array of these values and the length of the
#column. the processing of the column involves transforming it
#into unique values before returning any of the aformentioned
#variables
def proc_col(col): 
    uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx.get(x,1) for x in col]),\
    len(uniq)
    
#used in implicit feedback models.
#return users and items into a combined transposed (human readable)
#numpy array index
#eg. idx[13] = [16, 78]: index position 13 contains a combination of
#a user of id 16 and an item of id 78 (user 16 has rated item 78)
def rating_idx(users, items):
    idx = [users, items]
    idx = np.array(idx).transpose()
    return idx

#returns a tuple of indices of extinct users and items of a training
#set when compared to a testing set
def get_untrained(test_dataset, train_dataset):
    
    users = np.array([], dtype=int)
    items = np.array([], dtype=int)
    
    for i in test_dataset.data['userId'].unique():
        if i not in train_dataset.data['userId'].values:
            users = np.append(users, i)
    
    for i in test_dataset.data['movieId'].unique():
        if i not in train_dataset.data['movieId'].values:
            items = np.append(items, i)
    
    return users, items

#split dataframe into train and test sets
def split(data, method=2, value=None):
    try:
        #determined by testing set size
        if method == 1:
            if value == None:
                test = data.sample(n=1000)
            else:
                test = data.sample(n=value)
        #determined by split percentage
        elif method == 2:
            if value == None:
                test = data.sample(frac=0.1)
            else:
                test = data.sample(frac=value)
        train = data.drop(test.index)
        return train, test
    except Exception:
        print('Invalid method, incorrect dataset input, \
invalid testing set size or slicing \
percentage. Program will exit to avoid \
further errors.')
        sys.exit()