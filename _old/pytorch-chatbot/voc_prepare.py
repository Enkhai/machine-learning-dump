#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:02:26 2019

@author: ceyx
"""

import unicodedata
import re #regular expressions

from vocabulary import Voc

MAX_LENGTH = 10  #maximum sentence length to consider
MIN_COUNT = 3 #minimum word count for trimming

#convert unicode string to ascii
def unicodeToAscii(s):
    return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')

#lowercase, trim, remove non-letter
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

#reads query-response pairs and returns a Voc object
def readVocs(datafile, corpus_name):
    #read the file and split it into lines
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    #split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

#checks if query-response pairs are below maximum word length
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

#return pairs of only maximum or less word length
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

#creates a populated vocabulary and a pairs list from file
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    #read file and make vocabulary and conversation pairs
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    #fill the vocabulary
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    #return the vocabulary and the pairs
    print("Counted words:", voc.num_words)
    return voc, pairs

#trims words appearing less than the minimum trimming threshold
def trimRareWords(voc, pairs):
    #trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    #filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs