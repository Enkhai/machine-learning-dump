#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 12:36:26 2018

@author: ceyx
"""
#torch.nn.parallel.DistributedParallel has been proposed by the
#literature as a better and tested alternative for distributing
#and training the model in a parallel way.
#needs to be furtherly examined and tested

#import torch.utils.data as data
from sklearn.utils import shuffle

#consider transfering function to data_encode
def dataset_partition(data, num_partitions):
    data = shuffle(data)
    frac = 1/num_partitions
    partitions = ()
    for i in range(num_partitions):
        partition = data.sample(frac=frac)
        data.drop(partition.index)
        partitions = partitions + (partition,)
    return partitions