#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 04:44:36 2018

@author: ceyx
"""

#this code is experimental and doesn't work completely as
#intended
#refer to pytorch_distributed example for proper debugging

import os #OS
import torch #PyTorch
import torch.nn.functional as F #functions
import torch.distributed as dist #distributed
import torch.multiprocessing as mp #multiprocessing


#called 3rd.
#run functions:

#simple factorization module training. optimizer is Adam
#by default. prints training error per every epoch
def train_model_dist(rank, size, model, partitions, epochs=10, lr=0.01,
          wd=0.0, unsqueeze=False):
    print(f'\nRank {rank}, Making optimizer')
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,
                                 weight_decay=wd)
    
    partition = partitions[rank]
    users = torch.LongTensor(partition.userId.values).cuda()
    items = torch.LongTensor(partition.movieId.values).cuda()
    ratings = torch.FloatTensor(partition.rating.values).cuda()
    print(f'\nRank {rank}, Users, items, ratings were made')
    
    model = model.cuda()
    model.train()
    print(f'\nRank {rank}, Model is in training mode')
    print(f'\nRank {rank}, Model is {model}')
    
    torch.cuda.memory_allocated()
    
    if unsqueeze:
        ratings = ratings.unsqueeze(1)
        
    print(f'\nRank {rank}, Num of epochs: {epochs}')
    for i in range(epochs):
        print(f'\nRank {rank}, Epoch no {i}')
        y_hat = model(users, items)
        print(f'\nRank {rank}, Prediction done')
        loss = F.mse_loss(y_hat, ratings)
        print(f'\nRank {rank}, Loss computed')
        print(f'\nRank {rank}, Loss: {loss}')
        optimizer.zero_grad()
        print(f'\nRank {rank}, Gradients zeroed')
        loss.backward()
        print(f'\nRank {rank}, Computed new gradients')
        optimizer.step()
        print(f'\nRank {rank}, Computed new model weigths')
        print(f'Training loss at epoch {i}: {loss.item()} ')

#test loss evaluation    
def test_loss(model, test_set, unsqueeze=False):
    
    model = model.cuda()
    model.eval()
    
    users = torch.LongTensor(test_set.userId.values).cuda()
    items = torch.LongTensor(test_set.movieId.values).cuda()
    ratings = torch.FloatTensor(test_set.rating.values).cuda()
    if unsqueeze:
        ratings = ratings.unsqueeze(1)
    y_hat = model(users, items)
    loss = F.mse_loss(y_hat, ratings)
    print(f'\nTest loss: {loss.item()}')
    
#debug function
def test_run(*args):
    print('\nTest run')
    for arg in args:
        print(arg)
    kdba = torch.LongTensor([1,5,2]).cuda()
    print(kdba)


#called 2nd.
#standard distributed initialization process and 
#job attributing
def init_processes(rank, size, fn, backend='gloo', *args):
    """ Initialize the distributed environment. """
    os.environ['WORLD_SIZE']=str(size)
    os.environ['RANK']=str(rank)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    print('\nInitializing process: ', rank)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, *args)

#called 1st.
#fork, start and join processes
def train_dist(size, *args):
    #size = 2
    processes = []
    for rank in range(size):
        print('\nCreating process: ', rank)
        p = mp.Process(target=init_processes, args=(rank, size, train_model_dist, 'gloo', *args))
        #p = mp.Process(target=init_processes, args=(rank, size, test_run, 'gloo', *args))
        #p = mp.Process(target=test_run, args=(rank, size, train_model_dist, 'gloo', *args))
        print('\nStarting process: ', rank)
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print('\nProcesses joined')