#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 21:08:28 2019

@author: ceyx
"""
#importing

#import libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

#import methods & functions
from loaders import printLines, loadLines, loadConversations, extractSentencePairs
from voc_prepare import loadPrepareData, trimRareWords
from vocdata_prepare import batch2TrainData
from models import EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder
from training import evaluateInput

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")



#make the conversations file

#folder
corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)

#test
printLines(os.path.join(corpus, "movie_lines.txt"))

#the new conversations file
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

delimiter = '\t'
#unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

#lines, conversations, field IDs
lines = {}
conversation = []
LINE_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
CONVERSATION_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

#load the lines
print("\nLoading lines...")
lines = loadLines(os.path.join(corpus, "movie_lines.txt"), LINE_FIELDS)
#load the conversations
print("\nLoading conversations...")
conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"), lines, CONVERSATION_FIELDS)

#make/write the file
print("\nWriting conversation pairs to file...")
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    #write the conversation pairs
    for pair in extractSentencePairs(conversations):
        writer.writerow(pair)

#test
print("\nSample lines from file:")
printLines(datafile)



#make a vocabulary and a conversation pairs list from the formatted lines file

#assemble voc and pairs
save_dir = os.path.join("data", "save")
print("\nAssembling vocabulary data & conversation pairs...")
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)

#trim rare words (threshold = 3)
print("\nTrimming rare words. Appearance threshold is 3")
pairs = trimRareWords(voc, pairs)

#test
print("\npairs:")
for pair in pairs[:10]:
    print(pair)



small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)



# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')



# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, voc)