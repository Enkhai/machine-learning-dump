#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 22:18:28 2019

@author: ceyx
"""

#prints n lines of a file
def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

#returns a dictionary of the lines
def loadLines(fileName, fields):
    lines = {} #lines dictionary
    #open lines file
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f: #for each line
            values = line.split(" +++$+++ ") #split line to values
            lineObj = {} #line object. a dictionary of line attributes
            for i, field in enumerate(fields): #for each field
                lineObj[field] = values[i] #append field value to line object
            lines[lineObj['lineID']] = lineObj #when done, append line to lines dict
    return lines

#returns the list of conversations
def loadConversations(fileName, lines, fields):
    conversations = [] #conversations list
    with open(fileName, 'r', encoding='iso-8859-1') as f: #open conversations file
        for line in f: #for each line
            values = line.split(" +++$+++ ") #split line to values
            convObj = {} #conversation object. a dictionary of conversation attributes
            for i, field in enumerate(fields): #for each field
                convObj[field] = values[i] #append field value to conversation object
            lineIds = eval(convObj["utteranceIDs"]) #get the list of lines IDs for each conversation
            convObj["lines"] = [] #conversation lines list. lines values
            for lineId in lineIds: #for each line
                convObj["lines"].append(lines[lineId]) #append the value of the line
            conversations.append(convObj) #when done, append conversation to conversations list
    return conversations

#extracts pairs of questions and answers
def extractSentencePairs(conversations):
    qa_pairs = [] #question-answer pairs
    for conversation in conversations: #for each conversation
        #for each conversation line
        for i in range(len(conversation["lines"]) -1): #(ignore the last line - no answer)
            inputLine = conversation["lines"][i]["text"].strip() #question
            targetLine = conversation["lines"][i+1]["text"].strip() #answer
            #filter samples when one of the lists is empty
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine]) #append the question-answer pair
    return qa_pairs