# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 02:20:35 2018

@author: Ceyx
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist
#import torch.nn.parallel

#factorization module training. optimizer is Adam
#by default. options for distributed and manual distributed training,
#cuda training and high precision mode (average weights after finish) 
#prints training error per every epoch
def train_model(model, partition, epochs=10, lr=0.01, 
                wd=0.0, unsqueeze=False, cuda=False, manual=False,
                dist=False, high_precision=False, queue=None, rank=None, size=None):
    rank_prt = f"Rank {rank}: "
        
    if dist == True:
        if manual == True:
            if cuda == True:
                model = model.cuda()
            else:
                model = model.cpu()
            print(f'{rank_prt}Training is distributed! Made a copy of the model! (Manual distributed mode)')
        else:
            if cuda == True:
                model = torch.nn.parallel.DistributedDataParallel(model)
            else:
                model = torch.nn.parallel.DistributedDataParallelCPU(model)
            print(f'{rank_prt}Training is distributed! Set the model to distributed parallel mode!')
            #big change in memory usage increase when training is distributed
        partition = partition[rank]
        print(f'{rank_prt}Training dataset partitioned successfully!')
    elif dist == False and cuda == True:
        model = torch.nn.DataParallel(model)
                
    users = torch.LongTensor(partition.userId.values)
    items = torch.LongTensor(partition.movieId.values)
    ratings = torch.FloatTensor(partition.rating.values)
    print(f'{rank_prt}Users, items, ratings have been created')
    
    if cuda == True:
        users = users.cuda()
        items = items.cuda()
        ratings = ratings.cuda()
        print(f'{rank_prt}Users, items, ratings have been cast to cuda')
        
    #optimizer must be created after the model has been set to a definite
    #location
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,
                                 weight_decay=wd)
    print(f'{rank_prt}Created optimizer')
        
    model.train()
    print(f'{rank_prt}Model is in training mode')
    if unsqueeze:
        ratings = ratings.unsqueeze(1)
        
    print('')
    for i in range(epochs):
        y_hat = model(users, items)
        #re-initializing a model for the prediction causes a big change
        #in memory usage!!! (2 tasks = 2x2 models = 4 models in gpu!!)
        loss = F.mse_loss(y_hat, ratings)
        optimizer.zero_grad()
        loss.backward()
        if dist == True and manual == True:
            average(model) #no memory usage increase
        optimizer.step()
        print(f'{rank_prt}Training loss at epoch {i}: {loss.item()} ')
    
    #high precision mode. model parameters are also averaged
    #after the end of the training
    if high_precision == True and dist == True:
        average(model, param=False, w=True)
    
    #when the model is a DistributedParallel module there is no need for this
    if manual == True and rank == 0:
        #model must be cast to cuda() to be read by the queue
        #for some reason...
        #when model is in cpu(), queue.get() returns
        #EOF error
        model = model.cpu() #recast to cpu...
        model = model.cuda() #and recast to cuda...
        queue.put(model)
        print(f'\n{rank_prt}Model has been put into queue\n')
        
#model parameter and/or weight averaging
#distributed averaging has no effect on memory
def average(model, param=True, w=False):
    size = float(dist.get_world_size())
    if param == True:
        for param in model.parameters():
            #added a custom parameter in SVD that does not change
            #(no gradient)!!!
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
                param.grad.data /= size
                
    if w == True:
        for param in model.parameters():
            dist.all_reduce(param.data, op=dist.reduce_op.SUM)
            param.data /=size
    
#calculates loss error between the prediction and the testing
#set. default method of calculation is mean squared error
def test_loss(model, test_set, unsqueeze=False,
              cuda=False):
    model.eval()
    
    users = torch.LongTensor(test_set.userId.values)
    items = torch.LongTensor(test_set.movieId.values)
    ratings = torch.FloatTensor(test_set.rating.values)
    
    if cuda == True:
        users = users.cuda()
        items = items.cuda()
        ratings = ratings.cuda()
    
    if unsqueeze:
        ratings = ratings.unsqueeze(1)
    y_hat = model(users, items)
    loss = F.mse_loss(y_hat, ratings)
    print(f'\nTest loss: {loss.item()}')
    
#sets the model weights for items and users not included 
#in the training set (untrained and unchanged weights)
#to the mean item or user weight.
#a classic cold start problem solution
def calculate_model_untrained_weights(model, untrained_users, untrained_items, trained_users, trained_items, cuda=False):
    
    user_trained_indices = torch.LongTensor(trained_users)
    item_trained_indices = torch.LongTensor(trained_items)
    
    #the rest of the tensors below will also be cuda if the indices are as well
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