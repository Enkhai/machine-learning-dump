# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 18:10:34 2018

@author: Ceyx
"""

#neighbourhood-based algorithm

#part 1: importing

#libraries
import numpy as np #numpy
import pandas as pd #pandas
from pathlib import Path #path
import time #time
import matplotlib.pyplot as plt #plotting
import seaborn as sns #data visualization based on matplotlib

#methods
from helper import train_test_split, fast_similarity, predict_topk_nobias, get_mse

#part 2: preparing data

PATH = Path('..\..\Datasets\ml-100k')

#file does not contain headers
#we append them ourselves
names = ['user_id', 'item_id', 'rating', 'timestamp']
#data seperated by tabs, set headers to the names list
df = pd.read_csv(PATH/'u.data', sep='\t', names=names)
print(df.head()) #get the first 5

#number of unique users, items
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print(f'\n{n_users} users')
print(f'{n_items} items')

#make the ratings matrix
ratings = np.zeros((n_users, n_items))
#make the ratings matrix
#we map user/item IDs to user/item indices by removing the 
#"Python starts at 0" offset between them
for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]
print(f"\n {ratings}")

#sparsity
#(get non-zero ratings)
sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0]*ratings.shape[1])
#percentage
sparsity *= 100
print(f'\nRatings sparsity: {sparsity}%')

#make the training and test sets
train, test = train_test_split(ratings)

#part 3: make the similarities table

start = time.time()
user_similarity = fast_similarity(train)
item_similarity = fast_similarity(train, kind = 'item')
end = time.time()
print(f'\nTime elapsed for measuring cosine \
similarity with fast method: {end-start}')

#print the first four in the item similarity table
print (f'\nitem_similarity[:4, :4]: {item_similarity[:4, :4]}')

#part 4: make the predictions table

start = time.time()
item_prediction = predict_topk_nobias(train, item_similarity, kind='item')
user_prediction = predict_topk_nobias(train, user_similarity)
end = time.time()
print(f'\nTime elapsed for predicting \
similarity with Top-k method with bias: {end-start}')

#part 5: testing

print(f'\nUser-based CF MSE: {get_mse(user_prediction, test)}')
print(f'Item-based CF MSE: {get_mse(item_prediction, test)}')

#let's try a different amount of top-k neighbours    
#test (really slow). 50 seems to be optimal amount of top-k neighbors
    
k_array = [5, 15, 30, 50, 100, 200]
user_train_mse = []
user_test_mse = []
item_test_mse = []
item_train_mse = []

for k in k_array:
    #top k predictions
    user_pred = predict_topk_nobias(train, user_similarity, kind='user', k=k)
    item_pred = predict_topk_nobias(train, item_similarity, kind='item', k=k)
    
    #sum the mse for each:
    #user
    user_train_mse += [get_mse(user_pred, train)]
    user_test_mse += [get_mse(user_pred, test)]
    #item
    item_train_mse += [get_mse(item_pred, train)]
    item_test_mse += [get_mse(item_pred, test)]
    
#make a seaborn set
sns.set()

#pick a color palette to draw from (Set2 palette, 2 colors selected)
pal = sns.color_palette("Set2", 2)

plt.figure(figsize=(8,8)) #8x8
#draw the following
#alpha (intensity) = 0.5
plt.plot(k_array, user_train_mse, c=pal[0], label='User-based train', alpha=0.5, linewidth=5)
plt.plot(k_array, user_test_mse, c=pal[0], label='User-based test', linewidth=5)
plt.plot(k_array, item_train_mse, c=pal[1], label='Item-based train', alpha=0.5, linewidth=5)
plt.plot(k_array, item_test_mse, c=pal[1], label='Item-based test', linewidth=5)
#draw the legend at the best location, fonts: 20px
plt.legend(loc='best', fontsize=20)
#make the ticks as following
plt.xticks(fontsize=16);
plt.yticks(fontsize=16);
#and the labels
plt.xlabel('k', fontsize=30);
plt.ylabel('MSE', fontsize=30);
plt.show()