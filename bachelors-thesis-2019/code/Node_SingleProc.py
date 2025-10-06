#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 20:02:51 2018

@author: ceyx
"""
#part 1: importing

from training import train_model, test_loss, set_model_untrained_weights

import numpy as np #numpy
import time #time
import os #operating system
import sys #system
import torch #PyTorch
import torch.distributed as dist #distributed

#torch.set_num_threads(os.cpu_count())

#reproducibility
torch.manual_seed(0)
np.random.seed(0)

#part 2: preparing data

#load the datasets (torch.utils.data.Dataset)
train_dataset = torch.load('movielens_training_set.pkl')
test_dataset = torch.load('movielens_test_set.pkl')
        
print(f'\nTraining set: {train_dataset.data}')
print(f'\nTesting set: {test_dataset.data}')
    
#part 3: modeling

#load the model
model = torch.load("model.pkl").cuda()
print(f'\nModel: {model}')
    
#part 4: training    

backend = 'gloo'
node = int(sys.argv[1])
world_size = int(sys.argv[2])
    
#os.environ['MASTER_ADDR'] = '192.168.1.4'
os.environ['MASTER_ADDR'] = '195.251.122.92'
#os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['WORLD_SIZE'] = str(world_size)
os.environ['RANK'] = str(node)
os.environ['GLOO_SOCKET_IFNAME'] = 'enp3s0'
dist.init_process_group(backend, init_method="tcp://195.251.122.92:29500", world_size=world_size, rank=node)
#dist.init_process_group(backend, world_size=world_size, rank=node)
              
start = time.time()
train_model(model, train_dataset, cuda=True, distributed_mode=True, rank=node, world_size=world_size)
end = time.time()
            
print(f'\nTime elapsed for training: {end-start} sec')
    
#average model weights for untrained users/items 
set_model_untrained_weights(model, train_dataset, test_dataset, cuda=True)
    
#part 5: testing
    
test_loss(model, test_dataset, cuda=True)
