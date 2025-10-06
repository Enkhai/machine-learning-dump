#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 12:14:48 2019

@author: ceyx
"""

import itertools
import torch

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

#matches words in a sentence with vocabulary indices and returns a list of indices, plus
#the end-of-sentence token to signal the ending of the sentence
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


#fills short sentences
def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

#makes a binary matrix (mask) symbolizing pads with 0 and characters with 1
def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

#returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l] #word indexes
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch]) #sentence lengths tensor
    padList = zeroPadding(indexes_batch) #pad the indexes list
    padVar = torch.LongTensor(padList) #padded indexes tensor
    return padVar, lengths

#returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l] #word indexes
    max_target_len = max([len(indexes) for indexes in indexes_batch]) #find max sentence length
    padList = zeroPadding(indexes_batch) #pad the indexes list
    #make a character mask from the padded indexes list
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList) #padded indexes tensor
    return padVar, mask, max_target_len

#returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len