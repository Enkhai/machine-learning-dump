# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 06:35:55 2018

@author: Ceyx
"""
import torch
import torch.nn as nn
import numpy as np
#import torchnet #this does not exist within the installation
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import sys
import pickle
from pathlib import Path
import pandas as pd
import asyncio
import queue
import threading

from data_encode import encode, partition, split
#from training_distributed import test_run
from datasets import CF_Dataset

#reproducibility
torch.manual_seed(0)
np.random.seed(0)



#multi-threading
'''
def constant_print(q):
    print("ahoy")
  
    while True:
        prt = q.get()
        if prt == ".end": break
        print(prt)
    
q = queue.Queue()            
thread = threading.Thread(target=constant_print, args=(q,))
thread.start()

#manager = mp.Manager()
#q = manager.Queue()
#pool = mp.Pool(processes=1)
#pool.apply_async(constant_print, args=(q,))

for i in range(8):
    q.put(i)
    
q.put(".end")

thread.join()
    
#loop = asyncio.get_event_loop()
#done = loop.run_until_complete(constant_print(q))   

#print_task = asyncio.create_subprocess_exec(print('yo'))
'''

#save the training and test set into files,
#initialize a dataset and a distributed datasampler and use
#a dataloader to iterate through the data
'''
#import data
PATH = Path("../Datasets/ml-latest-small")
data = pd.read_csv(PATH/"ratings.csv")

data = encode(data)

#dataset splitting
train_set, test_set = split(data)

train_dataset = CF_Dataset(train_set)
test_dataset = CF_Dataset(test_set)
        
print(f'\nTraining set: \n{train_dataset}')
print(f'\nTesting set: \n{test_dataset}')

fw = open('movielens_training_set.pkl', 'wb')
pickle.dump(train_dataset, fw)
fw = open('movielens_test_set.pkl', 'wb')
pickle.dump(test_dataset, fw)


os.environ['MASTER_ADDR'] = '127.0.01'
os.environ['MASTER_PORT'] = '29500'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
dist.init_process_group('gloo')
test_datasampler = DistributedSampler(test_dataset, num_replicas=1, rank=0)

test_dataloader = DataLoader(test_dataset, batch_size=100, num_workers=0, sampler=test_datasampler)

for batch_num, batch_sample in enumerate(test_dataloader):
    print(f"Batch number: {batch_num}, batch: \nuserId: {batch_sample['userId']},\n" + 
          f"movieId: {batch_sample['movieId']}, \nrating: {batch_sample['rating']}")

#encode the dataset into unique and contiguous values
#train_parts = partition(train_set, 3)
#print(train_parts)
'''

#pickling-unpickling
'''
import sys
sys.path.append('../')
import pickle
fr = open('shared.pkl', 'rb')
iteration = pickle.load(fr)
    
fw = open('shared.pkl', 'wb')
pickle.dump(1, fw)
'''

#broadcast test
'''
os.environ['MASTER_PORT'] = '29500'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['WORLD_SIZE'] = '3'
os.environ['RANK'] = '0'
dist.init_process_group(backend = 'gloo', world_size = 3, rank = 0)
ah = torch.LongTensor([1])
dist.broadcast(ah, 0)

os.environ['MASTER_PORT'] = '29500'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['WORLD_SIZE'] = '3'
os.environ['RANK'] = '1'
dist.init_process_group(backend = 'gloo', world_size = 3, rank = 1)
oof = torch.LongTensor([2])
dist.broadcast(oof, 0)
print(oof)

os.environ['MASTER_PORT'] = '29500'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['WORLD_SIZE'] = '3'
os.environ['RANK'] = '2'
dist.init_process_group(backend = 'gloo', world_size = 3, rank = 2)
aaf = torch.LongTensor([3])
dist.broadcast(aaf, src=0)
print(aaf)
'''

#broadcasting for distributed nodes
'''
from models import SVD

#init
os.environ['MASTER_PORT'] = '29500'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['WORLD_SIZE'] = '2'
os.environ['RANK'] = '0'
dist.init_process_group(backend = 'gloo', world_size = 2, rank = 0)
print(f"Rank {dist.get_rank()}")


#process 0
users = torch.LongTensor([3])
items = torch.LongTensor([7])
mean = torch.FloatTensor([2.1])

#process 1
users = torch.LongTensor([1])
items = torch.LongTensor([1])
mean = torch.FloatTensor([1])

req = dist.broadcast(tensor=users, src=0, async_op=True)
dist.broadcast(items, 0)
dist.broadcast(mean, 0)
req.wait()

print(f'Users: {users}')
print(f'Items: {items}')
print(f'Mean: {mean}')

model = SVD(int(users),int(items),int(mean))
model = nn.parallel.DistributedDataParallel(model)

print(f"Rank 0: {model}")

for param in model.parameters():
    print(f"Parameter:{param}\n")
'''

#memory allocation
'''
for i in range(5):
    x = torch.LongTensor([2,i]).cuda()
    print(f'Allocated cuda memory: {torch.cuda.memory_allocated()}')
'''

#cuda properties
'''
torch.cuda.get_device_name(0)
torch.cuda.get_device_capability(0)
torch.cuda.get_device_properties(0)
'''

#call by object
'''
#call by object test and example
x = 9

def ref_demo(x):
    x+=42
    #vs
    #x = x + 42
    print("x=",x," id=",id(x))

print("x=",x," id=",id(x))
'''

#distributed training test
'''
#cuda testing with multiprocessing and distributed

#torch.cuda.init()
#tensor_out = torch.ones(2).cuda()
#let's take a look into that tensor
#print('What is the tensor\'s ID in the main program?...')
#print(id(tensor_out))
#print()
#if __name__ != '__main__':
#    print('Watch the magic...!')
#    print('I\'m using the spawn method and this isn\'t the \
#main program but only a copy of it!!!')
#    print('This is why I have a brand new different tensor!!!')
#    print()


def all_reduce(rank, size, tensor):
    print('Hello from task ', rank, '!')
    #create a new group with the following
    #processes
    #Notice: all processes belonging in the
    #distributed job MUST be entered, even if
    #they are not to be used.
    #Groups should also be created in the same
    #order in all processes
    group = dist.new_group([r for r in range(size)])
    #make a simple tensor
    #tensor = tensor_out
    print('Before:')
    print('Rank ', rank, ' has data ', tensor)
    print(f'Rank {rank}: What is that tensor\'s ID before...?')
    print(id(tensor))
    #everybody in the group do an all reduce
    #operation summing up the tensors received
    #by everyone else
    #everybody writes on a shared cuda allocation
    #TAKE CARE!!!
    dist.all_reduce(tensor=tensor[rank], op=dist.reduce_op.SUM, group=group)
    print('After:')
    print('Rank ', rank, ' has data ', tensor)
    print(f'Rank {rank}: What is that tensor\'s ID after...?')
    print(id(tensor))
    

def init_processes(rank, size, fn, tensor, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = str(size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend, world_size=size, rank=rank)
    fn(rank, size, tensor)    
    
#stuff done by the main process must be exclusive to the
#main. spawned processes must NOT repeat after their
#parent!!!
if __name__ == "__main__":
    mp.set_start_method('spawn')
    size = 2
    processes = []
    #what happens if i have one shared tensor?...
    tensor_out = torch.ones(2).cuda()
    for rank in range(size):
        print('Initializing task ', rank)
        #p = mp.Process(target=init_processes, args=(rank, size, test_run))
        p = mp.Process(target=init_processes, args=(rank, size, all_reduce, tensor_out))
        print('Run process of task ', rank)
        p.start()
        processes.append(p)
            
    print('\nActive processes before joining: ', mp.active_children())
    print('Joining ', len(mp.active_children()) ,' processes')
    for p in processes:
        p.join()
                
    print('Active processes after joining: ', mp.active_children())
    
    #better than file saving????!!!!!! 
    #TEST THIS FURTHER!!
    os.environ['TASK_CALLS']='0'
    
    #BUT.
    #how do i pass the model by importing?...
    
    #import sys
    #sys.path.insert(0, '/examples') #doesn't work exactly like that
    iteration = 0
    import pickle
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "shared.pkl"), "wb") as f:
    #fp = open('shared.pkl', "wb")
        pickle.dump(iteration, f)
    import examples.TEST_pytorch_distributed
'''

#SVD++ test
'''
from models import SVD, SVD_pp

#model testing
users = torch.LongTensor([3, 2, 5])
items = torch.LongTensor([6, 8, 12])
indices = np.random.randint(5, size=(2, 10)).transpose()

model_1 = SVD(6, 7, 3.2)
model_2 = SVD_pp(8, 15, indices, 3.2)

num_rated_items, rated_items, user_size = model_2.get_user_rated(users)

try:
    nums = (num_rated_items.data[1].item()**(-1/2))
except ZeroDivisionError:
    nums = 0

impl = model_2.item_implicit_emb(torch.LongTensor([np.asarray(rated_items[0]).tolist()])).sum(0)


if tuple(np.array(impl.size()).shape)[0] >=2 :
    impl = impl.sum(0)

num_rated_items, rated_items, user_size = model_2.get_user_rated(users)

torch.distributed._backend = torch.distributed.dist_backend.GLOO
torch.cuda.device(0)

oy = torch.LongTensor([4,7,1]).cuda()
ay = torch.LongTensor([12, 11, 5]).cuda()

eh = torch.ones(1).cuda()
yo = torch.ones(1).cuda()

ah = oy +ay

model_1.cuda()
#torch.nn.parallel.DistributedDataParallel needs process
#initialization
#model_1 = torch.nn.parallel.DistributedDataParallel(model_1)
'''