# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 06:35:55 2018

@author: Ceyx
"""
import torch
import torch.nn as nn
import numpy as np

#1. sparse tensor tutorial:

#indices, prwta stoixeia users, deytera items
i = torch.LongTensor([[2,0],[2,4],[2,5],[3,1],[4,2],[4,3],[6,0],[6,2]])
#values
v = torch.ones(8, dtype=torch.float64)
#prepei na kanoume transpose ta indices!!
sparse = torch.sparse.FloatTensor(i.t(), v, torch.Size([8,6]))


#2. random things

#ena tyxaio embedding
#emb(item_count)
emb = nn.Embedding(20,10)

#oi monadikoi users (prwta stoixeia)
sparse._indices()[0].unique()


#3. epistrofh num_rated_items kai rated_items
#(bl. SVD_pp, methodos get_user_rated)

#ari8mos monadikwn xrhstwn
size_0 = sparse._indices()[0].unique().size()
#o ari8mos twn monadikwn xrhstwn se integer
size = tuple(size_0)[0]
#ari8mos antikeimenwn p exei aksiologhsei o ka8e xrhsths
uniq_count = torch.LongTensor(size_0)
#indices antikeimenwn p exei ba8mologhsei o ka8e xrhsths
item_count = np.empty([size], dtype=list)
#iteration-xrhsimo gia to indexing
iteration = torch.LongTensor([0])

#gia ka8e xrhsth
for user in sparse._indices()[0].unique():
    #posa antikeimena ba8mologhse??
    uniq = (sparse._indices()[0] == user.item()).sum()
    #apo poia kelia antiproswpeuetai o xrhsths?? px ta 2 
    #kelia 6 kai 7 aforoun eggrafes-ratings tou xrhsth 
    #panw se 2 antikeimena
    item_indices = (sparse._indices()[0] == user.item()).nonzero()
    #poia antikeimena antiproswpeuontai se ayta ta kelia??
    uniq_item = sparse._indices()[1].data[item_indices].squeeze().tolist()
    #pros8ese ston pinaka twn ari8mwn antikeimenwn gia ton
    #xrhsth [iteration] ton ari8mo twn antikeimenwn pou
    #ba8mologhse
    uniq_count.put_(iteration, uniq)
    #uniq_count = torch.cat((uniq_count, uniq))
    #pros8ese ston pinaka twn antikeimenwn pou ba8mologhse o
    #[iteration] xrhsths mia lista apo ta antikeimena tou
    item_count[iteration.numpy()[0]] = uniq_item
    #aykshse to iteration gia ton epomeno xrhsth!!
    iteration +=1
    
#num_rated_items :)
print(uniq_count)
print('')
#rated_items :)
print(item_count)

#4. xtisimo pinaka implicit data pros pros8esh sta barh
#tou pinaka paragontwn twn users gia ton teliko ypologismo.
#to emb pou dhmiourgh8hke parapanw einai enas 8ewrhtikos
#pinakas implicit barwn
#(bl. SVD_pp, telikos ypologismos methodou forward)

#kane to iteration ksana 0
iteration = torch.LongTensor([0])
#8eloume na kanoume prakseis. to apotelesma 8a apo8hkeytei
#se ena FloatTensor anagkastika!!
output = torch.FloatTensor()
#me mege8os oso einai kai to embedding!!
output = output.new_empty((size,10))
#gia ka8e xrhsth
for user in uniq_count:
    #(ari8mos antikeimenwn pou ba8mologhse o xrhsths)^-1/2
    #epi (to a8roisma tou pinaka paragontwn gia ka8e
    #antikeimeno pou ba8mologhse o xrhsths)
    output_item = (uniq_count.data[iteration].item()**(-1/2))*\
    emb(torch.LongTensor(np.asarray(item_count[iteration.numpy()[0]]))).sum(0)
   
    output.data[iteration] = output_item
    #output.put_(iteration, ...)
    iteration +=1
    
print(output)