#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 20:03:18 2018

@author: ceyx
"""
#part 1: importing

#coded methods
from training import train_model, test_loss, set_model_untrained_weights

#libraries
import numpy as np #numpy
import time #time
import os #operating system
import sys #system
import copy #copy
import torch #PyTorch
import torch.multiprocessing as mp #multiprocessing
import torch.distributed as dist #distributed

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
    
    #load the datasets (torch.utils.data.Dataset)
    train_dataset = torch.load('movielens_training_set.pkl')
    test_dataset = torch.load('movielens_test_set.pkl')
            
    print(f'\nTraining set: \n{train_dataset.data}\n\n')
    print(f'\nTesting set: \n{test_dataset.data}\n\n')
    
    #part 3: modeling

    #load the model
    model = torch.load('model.pkl').cuda()
    print(model)    

    #part 4: training
    
    first_task, last_task = eval(sys.argv[1])
    world_size = int(sys.argv[2])
        
    #train_args = [model, train_dataset]
    kw_train_args = {'distributed_mode':True, 'cuda':True}
    
    #parallel section      
    start = time.time() #start counter
    mp.set_start_method('spawn')
    processes = []
    print('\nTraining is distributed. One model copy per process\n\n')
    for rank in range(first_task, last_task+1):
        model = copy.deepcopy(model) #new model reference for each process
        train_args = [model, train_dataset] #reinstate the train arguments
        #arguments
        arguments = [rank, world_size, train_model]
        arguments.extend(train_args)
        #key-worded arguments
        kw_arguments = {}
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