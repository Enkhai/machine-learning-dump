#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 20:03:18 2018

@author: ceyx
"""

#NodeB: slave node

#NEEDS **EXTENSIVE** WORK
#NEEDS **EXTENSIVE** WORK
#NEEDS **EXTENSIVE** WORK
#NEEDS **EXTENSIVE** WORK

#part 1: importing

#coded methods
from split_selection import input_select, input_tasks
from data_encode import encode, split, partition, test_train_deviation
from models import SVD
from training import train_model, test_loss, calculate_model_untrained_weights

#libraries
#import numpy as np #numpy
import pandas as pd #pandas
from pathlib import Path #path
import time #time
import os #operating system
#import sys #system
#import pickle #pickle
#import torch #PyTorch
#import torch.nn as nn #Neural network module
#import torch.nn.functional as F #functions
import torch.multiprocessing as mp #multiprocessing
import torch.distributed as dist #distributed

#sets the number of OpenMP threads to be used by PyTorch to the maximum.
#the default is half of that amount
#torch.set_num_threads(os.cpu_count())
#torch.set_num_threads(1) #or just one thread...

#process initialization method
def init_processes(rank, size, fn, *args, backend='gloo', **kargs):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = str(size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend, world_size=size, rank=rank)
    fn(*args, rank=rank, size=size, **kargs)
    

#protection of resources and unrestrained spawning control
if __name__ == "__main__":
    
    #part 2: preparing data
    
    #import data
    PATH = Path("../Datasets/ml-latest-small")
    data = pd.read_csv(PATH/"ratings.csv")
    
    #encode the dataset into unique and contiguous values
    data = encode(data)
    
    raw_num_ratings = len(data.rating)
    raw_num_users = len(data.userId.unique())
    raw_num_items = len(data.movieId.unique())
    max_item_idx = data.movieId.max()
    sparsity = raw_num_ratings/(raw_num_users*raw_num_items)
    
    print(f'\nNumber of users: {raw_num_users}')
    print(f'Number of items: {raw_num_items}')
    print(f'Largest item index: {max_item_idx}')
    print(f'Number of ratings: {raw_num_ratings}')
    print(f'Dataset sparsity: {sparsity}')
    
    #average rating
    mean_rating = data.rating.mean()
    
    print(f'\nAverage rating: {mean_rating}\n')
    
    #splitting variables input
    #method, value = input_select()
    
    #print(f'\nMethod: {method}\nValue: {value}')
    
    #dataset splitting
    train_set, test_set = split(data)
        
    print(f'\nTraining set: {train_set}')
    print(f'\nTesting set: {test_set}')
    
    #part 3: modeling
    
    #initialize the model in cpu()
    #NOTE: be careful not to overload the gpu if it is not necessary
    model = SVD(raw_num_users, max_item_idx+1, mean_rating)
    #set the model to cuda for gpu training
    model = model.cuda()
    print(f'\nModel: {model}')
    
    #part 4: training
    
    #num_tasks = input_tasks()
    num_tasks = 1
    
    #partition the dataset for the number of tasks (batch size).
    #manual distributed parallel mode only
    #train_parts = partition(train_set, num_tasks)
    #print('\nTrain partitions: \n', train_parts)
    
    #declare arguments here. these are obligatory arguments
    # #1 model, #2 training dataset partitions(manual mode) or training set
    #train_args = [model, train_parts]
    train_args = [model, train_set]
    #declare keyworded arguments here.
    #options include:
    #epochs(10)(int): number of epochs, lr(0.01)(float): learning rate,
    #wd(0.0)(float): weight decay
    #unsqueeze(false)(boolean): add one more dimension to the ratings for the training,
    #manual(false)(boolean): manual distributed training mode - declare training partitions
    #tuple instead of training set
    #dist(false)(boolean): distributed training, cuda(false)(boolean): train model in cuda, 
    #high_precision(false)(boolean): average model parameters of all processes after training
    kw_train_args = {'manual':False, 'dist':True, 'cuda':True, 'high_precision':False}
    
    #parallel section      
    start = time.time() #start counter
    mp.set_start_method('spawn')
    #use this when start method is fork (default) to
    #share a cpu model with forked processes
    #model.share_memory()
    processes = []
    queue = mp.SimpleQueue()
    for rank in range(num_tasks, num_tasks+1):
        arguments = [rank, 2, train_model]
        arguments.extend(train_args)
        kw_arguments = {"queue":queue}
        kw_arguments.update(kw_train_args)
        p = mp.Process(target=init_processes, args=arguments, kwargs=kw_arguments)
        p.start()
        processes.append(p)
    
    #use this only when performing manual distributed training
    #model = queue.get()
    #model can only be returned in cuda() with distributed
    #training because of a bug...
    #model = model.cpu()

    for p in processes:
        p.join()
        
    end = time.time() #end counter
            
    print(f'\nTime elapsed for training: {end-start} sec')