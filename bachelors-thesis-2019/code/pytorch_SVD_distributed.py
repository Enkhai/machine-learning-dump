# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 16:10:05 2018

@author: ceyx
"""

#part 1: importing

#coded methods
from split_selection import input_select, input_tasks
from data_encode import encode, split
from models import SVD
from datasets import CF_Dataset
from training import train_model, test_loss, set_model_untrained_weights

#libraries
import numpy as np #numpy
import pandas as pd #pandas
from pathlib import Path #path
import time #time
import os #operating system
#import sys #system
import copy
#import pickle #pickle
import torch #PyTorch
#import torch.nn as nn #Neural network module
#import torch.nn.functional as F #functions
import torch.multiprocessing as mp #multiprocessing
import torch.distributed as dist #distributed

#sets the number of OpenMP threads to be used by PyTorch to the maximum.
#the default is half of that amount
#torch.set_num_threads(os.cpu_count())
#torch.set_num_threads(1) #or just one thread...

#reproducibility
torch.manual_seed(0)
np.random.seed(0)

#process initialization method
def init_processes(rank, size, fn, *args, backend='gloo', **kargs):
    #os.environ['MASTER_ADDR'] = '192.168.1.5'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = str(size)
    os.environ['RANK'] = str(rank)
    #os.environ['GLOO_SOCKET_IFNAME'] = 'enp3s0'
    #dist.init_process_group(backend, init_method="tcp://192.168.1.5:29500", world_size=size, rank=rank)
    dist.init_process_group(backend, world_size=size, rank=rank)
    fn(*args, rank=rank, world_size=size, **kargs)
    

#protection of resources and unrestrained spawning control
if __name__ == "__main__":
    
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
    
    #torch.save(train_dataset, 'movielens_training_set.pkl')
    #torch.save(test_dataset, 'movielens_test_set.pkl')
    
    #part 3: modeling
    
    #initialize the model in cpu()
    #NOTE: be careful not to overload the gpu if it is not necessary
    model = SVD(num_users, num_items, mean_rating)
    #set the model to cuda for gpu training
    model = model.cuda()
    print(f'Model initialized: {model}')
        
    #part 4: training
    
    num_tasks = input_tasks()
    
    #obligatory training arguments (train_args):
    # #1 model, #2 training dataset
    #train_args = [model, train_dataset]

    #keyworded training arguments (kw_train_args).
    #options include:
    #epochs(10)(int): number of epochs, lr(0.01)(float): learning rate,
    #wd(0.0)(float): weight decay
    #unsqueeze(false)(boolean): add one more dimension to the ratings for the training,
    #cuda(false)(boolean): train model in cuda,
    #manual(false)(boolean): manual distributed training mode - model gradients are manually
    #averaged across all processes/nodes
    #distributed_mode(false)(boolean): distributed training, 
    #high_precision(false)(boolean): average model parameters of all processes after training,
    #rank(None)(int): distributed training mode only - indicates the rank of the running process,
    #world_size(None)(int): distributed training mode only - indicates the number of tasks/processes
    #participating in the training process
    kw_train_args = {'distributed_mode':True, 'cuda':True}
    
    #parallel section      
    start = time.time() #start counter
    mp.set_start_method('spawn')
    #use this when start method is fork (default) to
    #share a cpu model with forked processes
    #model.share_memory()
    processes = []
    print('Training is distributed. One model copy per process')
    for rank in range(num_tasks):
        model = copy.deepcopy(model) #new model reference for each process
        train_args = [model, train_dataset] #reinstate the train arguments
        #arguments
        arguments = [rank, num_tasks, train_model]
        arguments.extend(train_args)
        #key-worded arguments
        kw_arguments = {'wd':0.00028, 'epochs':100}
        kw_arguments.update(kw_train_args)
        #init
        p = mp.Process(target=init_processes, args=arguments, kwargs=kw_arguments)
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
        
    end = time.time() #end counter
                
    print(f'\nTime elapsed for training: {end-start} sec')
    
    #average model weights for untrained users/items 
    set_model_untrained_weights(model, train_dataset, test_dataset, cuda=True)
    
    #part 5: testing
    
    test_loss(model, test_dataset, cuda=True)