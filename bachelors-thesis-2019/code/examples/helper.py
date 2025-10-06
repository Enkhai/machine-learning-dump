# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 16:56:47 2019

@author: Ceyx
"""
import numpy as np
from sklearn.metrics import mean_squared_error #mse

#|----------------------------------------------------------------------------------------
#dataset splitting methods

#split dataset to training and test set
def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    #for each user
    for user in range(ratings.shape[0]):
        #select 10 random ratings of a user for an item,
        #without replacement
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                        size=10,
                                        replace=False)
        #exclude those ratings from the training set
        train[user, test_ratings] = 0.
        #append them to the test set
        test[user, test_ratings] = ratings[user, test_ratings]
    #assert (test) if the training and test sets contain
    #common ratings (they should not)
    assert(np.all((train*test == 0)))
    return train, test #return the test and training set arrays

#|----------------------------------------------------------------------------------------
#cosine similarity methods

#very slow. not recomended
def slow_similarity(ratings, kind='user'):
    #user case
    if kind == 'user':
        axmax = 0
        axmin = 1
    #item case
    elif kind == 'item': 
        axmax = 1
        axmin = 0
        #make the similarity matrix
        #(item*item or user*user depending on the case)
    sim = np.zeros((ratings.shape[axmax], 
                     ratings.shape[axmax]))
    #for every user or item
    for u in range(ratings.shape[axmax]):
        #for every user or item (again)
        for uprime in range(ratings.shape[axmax]):
            #sum factors
            rui_sqrd = 0.
            ruprimei_sqrd = 0.
            #for each column or row (depending on kind)
            for i in range(ratings.shape[axmin]):
                #numerator
                sim[u, uprime] = ratings[u, i] * ratings[uprime, i]
                #denominator root sums
                rui_sqrd += ratings[u,i] ** 2
                ruprimei_sqrd += ratings[uprime, i] ** 2
            #fill the similarity matrix for user or item
            sim[u, uprime] /= np.sqrt(rui_sqrd*ruprimei_sqrd)
    return sim
    
#fast    
#epsilon: very small number for handling divided-by-zero errors
def fast_similarity(ratings, kind='user', epsilon=1e-9):
    #user case
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    #item case (transposed case scenario)
    elif kind == 'item' :
        sim = ratings.T.dot(ratings) + epsilon
    #make the norms
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim/norms/norms.T)
 
 #|----------------------------------------------------------------------------------------   
#prediction methods
    
#normalized sum of cosine similarities (weights)
#multiplied by the ratings

#slow simple
def predict_slow_simple(ratings, similarity,
                        kind = 'user'):
    #make the prediction table
    #(same shape as the ratings table)
    pred = np.zeros(ratings.shape)
    #user case
    if kind == 'user':
        #for each user
        for i in range(ratings.shape[0]):
            #for each user (again)
            for j in range(ratings.shape[1]):
                #make the prediction
                pred[i,j] = similarity[i, :].dot(\
                    ratings[:,j])/np.sum(\
                           np.abs(similarity[i,:]))
   #item case
    elif kind == 'item':
        #for each user
        for i in range(ratings.shape[0]):
            #for each user (again)
            for j in range(ratings.shape[1]):
                #make the prediction (in reverse)
                pred[i,j] = similarity[j, :].dot(\
                    ratings[i,:].T)/np.sum(\
                           np.abs(similarity[j,:]))
        
#fast simple
def predict_fast_simple(ratings, similarity,
                        kind = 'user'):
    #user case
    if kind == 'user':
        #dot product of similarities (weights)
        #multiplied with the ratings divided by
        #the sum of the absolute values of the
        #cosine similarities (weights)
        return similarity.dot(ratings)/\
               np.array([np.abs(similarity)\
                         .sum(axis=1)]).T
    #item case
    elif kind == 'item':
        #the same but transposed
        return ratings.dot(similarity)/\
               np.array([np.abs(similarity)\
                         .sum(axis=1)])
    
#top-k neighbours prediction
def predict_topk(ratings, similarity, kind='user', k=40):
    #make the prediction matrix
    pred = np.zeros(ratings.shape)
    #user case
    if kind == 'user':
        for i in range(ratings.shape[0]):
            #top k users
            top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
            for j in range(ratings.shape[1]):
                #make the prediction for the user
                pred[i,j] = similarity[i,:][top_k_users].dot(ratings[:,j][top_k_users])
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
    #item case
    if kind == 'item':
        for j in range(ratings.shape[1]):
            #top k items
            top_k_items = [np.argsort(similarity[:,j])[:-k-1:-1]]
            for i in range(ratings.shape[0]):
                #make the prediction for the item
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T) 
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))
    return pred

#prediction with bias
def predict_nobias(ratings,similarity, kind='user'):
    #user case
    if kind == 'user':
        #mean rating is bias
        user_bias = ratings.mean(axis=1)
        #calculate the ratings again based on bias
        ratings = (ratings - user_bias[:, np.newaxis].copy())
        #make the prediction, then add bias
        #(h()*h()^-1 golden rule)
        pred = similarity.dot(ratings)/np.array([np.abs(similarity).sum(axis=1)]).T
        pred += user_bias[:, np.newaxis]
    #item case
    elif kind == 'item':
        #same as before
        item_bias = ratings.mean(axis=0)
        ratings = (ratings - item_bias[np.newaxis, :]).copy()
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        pred += item_bias[np.newaxis, :]
    return pred

#top-k prediction with bias
def predict_topk_nobias(ratings, similarity, kind='user', k=40):
    #make the prediction matrix
    pred = np.zeros(ratings.shape)
    #user case
    if kind == 'user':
        #mean bias
        user_bias = ratings.mean(axis=1)
        #remove bias
        ratings = (ratings - user_bias[:, np.newaxis]).copy()
        #for each user
        for i in range(ratings.shape[0]):
            #get the top-k users
            top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
            for j in range(ratings.shape[1]):
                #make the prediction for top-k neighbours
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users]) 
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
        #add the bias
        #(h()*h()^-1 golden rule)
        pred += user_bias[:, np.newaxis]
        #item case
    if kind == 'item':
        #same as before
        item_bias = ratings.mean(axis=0)
        ratings = (ratings - item_bias[np.newaxis, :]).copy()
        for j in range(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:,j])[:-k-1:-1]]
            for i in range(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T) 
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items])) 
        pred += item_bias[np.newaxis, :]
    return pred

#|----------------------------------------------------------------------------------------
#test loss methods

#mean squared error
def get_mse(pred, actual):
    #make the prediction table for the given ratings
    pred = pred[actual.nonzero()].flatten()
    #make the evaluation table for the given ratings
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)