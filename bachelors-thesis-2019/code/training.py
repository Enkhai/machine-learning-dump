# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 02:20:35 2018

@author: Ceyx
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallelCPU, DistributedDataParallel
import threading
import queue

from data_encode import get_untrained
from misc import queue_iter_print

#factorization module training. optimizer is Adam
#by default. options for distributed and manual distributed training,
#cuda training and high precision mode (average weights after finish) 
#prints training error per every epoch
def train_model(model, train_dataset, epochs=10, lr=0.01, 
                wd=0.0, unsqueeze=False, cuda=False,
                distributed_mode=False, rank=None, world_size=None):
    
    rank_prt = f"Rank {rank}: "
    sampler = None

    #printing is assigned to a new thread to minimize training time by removing
    #I/O interrupts
    #a queue will handle the passing of the print string messages to the new thread
    q = queue.Queue()
    print_thread = threading.Thread(target=queue_iter_print, args=(q,))
    print_thread.start()
    
    #distributed mode
    if distributed_mode:
        q.put(f"{rank_prt}Creating distributed sampler")
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
                
        if cuda:
            model = DistributedDataParallel(model)
        else:
            model = DistributedDataParallelCPU(model)
        q.put(f"{rank_prt}Set the model to distributed data parallel")
            
    #non-distributed. set the model to DataParallel to increase training speed
    elif not distributed_mode and cuda:
        model = torch.nn.DataParallel(model)
        
    q.put(f"{rank_prt}Making Dataloader")
    dataloader = DataLoader(train_dataset, batch_size=5001, num_workers=4, sampler=sampler)
        
    #optimizer created after the model has been set to a definite location
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,
                                 weight_decay=wd)
    q.put(f'{rank_prt}Created optimizer')
        
    model.train()
    q.put(f'{rank_prt}Model is in training mode\n')        
        
    for i in range(epochs):
        
        if distributed_mode:
            sampler.set_epoch(i)
        
        for batch_idx, batch_sample in enumerate(dataloader):

            train_prt = f"Epoch {i}, batch {batch_idx}: "
            
            users = batch_sample["userId"].long()
            items = batch_sample["movieId"].long()
            ratings = batch_sample["rating"].float()
            
            if unsqueeze:
                ratings = ratings.unsqueeze(1)
            
            if cuda:
                users = users.cuda()
                items = items.cuda()
                ratings = ratings.cuda()
        
            y_hat = model(users, items) #prediction
            loss = F.mse_loss(y_hat, ratings) #loss
            optimizer.zero_grad() #zero gradients
            loss.backward() #update gradients
            optimizer.step() #step
            q.put(f'{rank_prt}{train_prt}Training loss: {loss.item()} ')
        
    q.put(".end") #signal the print function to return
    print_thread.join() #and close the thread
    
#calculates loss error between the prediction and the testing
#set. default method of calculation is mean squared error
def test_loss(model, test_dataset, unsqueeze=False,
              cuda=False):
    model.eval()
    
    #make dataloader to read test dataset. make batch size very large
    #to retrieve the whole test set
    dataloader = DataLoader(test_dataset, batch_size=1000000)
    test_set = next(iter(dataloader))
    
    users = test_set['userId']
    items = test_set['movieId']
    ratings = test_set['rating']
    
    if cuda == True:
        users = users.cuda()
        items = items.cuda()
        ratings = ratings.cuda()
    
    if unsqueeze:
        ratings = ratings.unsqueeze(1)
    y_hat = model(users, items)
    loss = F.mse_loss(y_hat, ratings)
    print(f'\nTest loss: {loss.item()}')
    return loss.item()
    
#sets the model weights for items and users not included 
#in the training set (untrained and unchanged weights)
#to the mean item or user weight.
#a classic cold start problem solution
def set_model_untrained_weights(model, train_dataset, test_dataset, cuda=False):
    
    untrained_users, untrained_items = get_untrained(test_dataset, train_dataset)
    
    user_trained_indices = torch.LongTensor(train_dataset.data['userId'].unique())
    item_trained_indices = torch.LongTensor(train_dataset.data['movieId'].unique())
    
    if cuda == True:
        user_trained_indices = user_trained_indices.cuda()
        item_trained_indices = item_trained_indices.cuda()
    
    user_trained_weights = torch.index_select(model.user_emb.weight, dim=0, index = user_trained_indices)
    item_trained_weights = torch.index_select(model.item_emb.weight, dim=0, index = item_trained_indices)

    user_trained_average = user_trained_weights.mean(0)
    item_trained_average = item_trained_weights.mean(0)
    
    for i in untrained_users:
        model.user_emb.weight[i] = user_trained_average
        
    for i in untrained_items:
        model.item_emb.weight[i] = item_trained_average