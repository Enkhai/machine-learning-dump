# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 23:27:42 2018

@author: Ceyx
"""
import torch #PyTorch
import torch.nn as nn #Neural network module
import numpy as np #Numpy
import sys #system

#two-layer net
class TwoLayerNet(nn.Module):
    
    #constructor
    def __init__(self, D_in=1000, H=100, D_out=10):
        super(TwoLayerNet, self).__init__() #call base constructor
        self.linear1 = nn.Linear(D_in, H) #linear layer 1
        self.linear2 = nn.Linear(H, D_out) #linear layer 2
        
    #forward function (prediction)
    def forward(self, idx):
        hidden_out=self.linear1(idx).clamp(min=0) #hidden layer output
        output_out = self.linear2(hidden_out) #output layer output
        return output_out

#SVD (singular value decomposition) model with bias and mean
#rating
class SVD(nn.Module):
    def __init__(self, num_users, num_items, mean, emb_size=100):
        super(SVD, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.user_emb_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.item_emb_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(0, 0.005)
        self.user_emb_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(0, 0.005)
        self.item_emb_bias.weight.data.uniform_(-0.01, 0.01)
        self.mean = nn.Parameter(torch.FloatTensor([mean]), False)
        
        
    def forward(self, u, v):
        U = self.user_emb(u)
        b_u = self.user_emb_bias(u).squeeze()
        I = self.item_emb(v)
        b_i = self.item_emb_bias(v).squeeze()
        return (I*U).sum(1) + b_u + b_i + self.mean
    
#SVD++ (singular value decomposition) model with bias and
#implicit feedback factors, mean rating of rated items and 
#a 2D rating index tensor indicating user ratings (userId)
#on items (itemId) to accomodate calculations on 
#implicit feedback
class SVD_pp(SVD):
    def __init__(self, num_users, num_items, ratingidx, mean, emb_size=100):
        super(SVD_pp, self).__init__(num_users, num_items, mean, emb_size)
        self.item_implicit_emb = nn.Embedding(num_items, emb_size)
        self.item_implicit_emb.weight.data.uniform_(-0.01, 0.01)
        self.ratingidx = torch.from_numpy(ratingidx.transpose()).type(torch.LongTensor)
        self.emb_size = emb_size
        
    def forward(self, u, v):
        U = self.user_emb(u)
        b_u = self.user_emb_bias(u).squeeze()
        self.I = self.item_emb(v)
        b_i = self.item_emb_bias(v).squeeze()
        num_rated_items, rated_items, user_size = self.get_user_rated(u)
        implicit = self.get_implicit(num_rated_items, rated_items, user_size)
        return (self.I*(U + implicit)).sum(1) + b_u + b_i + self.mean
    
    #returns a 1D tensor of the amount, a list of
    #the indices of items users u have rated based on the
    #rating index tensor declared during class construction
    #and the number of unique users
    def get_user_rated(self, u):
        #size of users
        size_0 = u.size()
        size = tuple(size_0)[0]
        
        #declare variables to be returned plus an iteration
        #variable
        num_rated_items = torch.empty(size, dtype=torch.int64)
        rated_items = np.empty([size], dtype=list)
        i = torch.LongTensor([0])
        index = 0

        #for every user
        for user in u:
            #number of rated items per user
            count = (self.ratingidx[0] == user.item()).sum()
            num_rated_items.put_(i, count)
            
            #an array of lists of rated item indices per user
            user_indices = (self.ratingidx[0] == user.item()).nonzero()
            item_list = self.ratingidx[1].data[user_indices].squeeze().tolist()
            rated_items[index] = item_list
            
            #increment for the next iteration
            i +=1
            index +=1
        
        return num_rated_items, rated_items, size
    
    def get_implicit(self, num_rated_items, rated_items, user_size):
        
        #iteration variables
        i = torch.LongTensor([0])
        index = 0
        
        #return variable (tensor)
        output = torch.empty((user_size, self.emb_size), dtype=torch.float32)

        for user in num_rated_items:
            #(number of items rated by the user)^-1/2
            #multiplied by (a tensor of the sum of each
            #factor of items rated by the user)
            
            #part no.1 of the equation
            try:
                n1 = (num_rated_items.data[i].item()**(-1/2))
            except ZeroDivisionError:
                n1 = 0
            
            temp = np.asarray(rated_items[index]).tolist()
            if not isinstance(temp, (list,)):
                temp = [temp]                
            
            #part no. 2 of the equation
            try:
                n2 = self.item_implicit_emb(torch.LongTensor(temp)).sum(0)
            except Exception:
                print('\nError calculating part 2 of the equation. System will exit\
                      to avoid further errors.')
                sys.exit()
            #sum factors if there are more than one items
            if (tuple(np.array(n2.size()).shape)[0]) >= 2:
                n2 = n2.sum(0)            
            output_item = n1*n2
            
            output.data[i] = output_item
            
            #increment for the next iteration
            i +=1
            index +=1
        
        return output