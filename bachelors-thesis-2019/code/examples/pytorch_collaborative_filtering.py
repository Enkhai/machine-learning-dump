# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 08:18:19 2018

@author: Ceyx
"""

#imports
from pathlib import Path #file path
import pandas as pd #pandas (data analyzer)
import numpy as np #numpy
import torch #torch
import torch.nn as nn #nn module
import torch.nn.functional as F #functions

#orise path
PATH = Path("J:\Shared folder\TEI\It has begun\Datasets\ml-latest-small")
print(f'\nDirectory of PATH: \n {list(PATH.iterdir())}')#list directory of path

##dwse mou pisw to ratings
#(to data einai typou pandas.Dataframe)
data = pd.read_csv(PATH/"ratings.csv")
data.head() #diabase mou tis 5 (default) prwtes grammes

#dinoume ena seed gia thn dhmiourgia tyxaiwn ari8mwn
np.random.seed(3)
#ftiakse ena array mege8ous oso einai to dataset
#(me times apo 0 ews 1) kai epelekse osa apo ayta
#einai mikrotera tou 0.8
#(epistrefei boolean array)
msk = np.random.rand(len(data)) < 0.8
#dwse mou pisw ta dedomena tou data gia ta opoia isxyei
#to msk (mask- praktika maska dedomenwn)
train = data[msk].copy() #training set
#to idio me to parapanw g dedomena p den isxyei h maska
val = data[~msk].copy()

#h parakatw me8odos epilegei ta monadika stoixeia mias
#sthlhs, ta epeksergazetai kai epistrefei pisw kapoies
#sxetikes times - diabase parakatw
def proc_col(col, train_col=None):
    #an yparxei training column 8a doulepseis me aythn
   if train_col is not None:
       #pare mono mia fora ta stoixeia pou den einai
       #idia apo to training column kai kane ta
       #contiguous (synexomenes times)
       uniq = train_col.unique()
   #alliws an den yparxei training column
   else:
       #pare mono mia fora ta stoixeia pou den einai
       #idia apo to col kai kane ta
       #contiguous (synexomenes times)
       uniq = col.unique()
   #ftiakse ena dictionairy me ta dedomena ths sthlhs
   #kai to index tou ka8e stoixeiou
   name2idx = {o:i for i,o in enumerate(uniq)}
   #epestrepse mou to parapanw dictionairy,
   #to dictionairy ksana se ena array kai
   #to plh8os twn stoixeiwn (tuple)
   return name2idx, np.array([name2idx.get(x,1) \
   for x in col]), len(uniq)
    
#h parakatw me8odos pairnei ena Dataframe (df) h
#ena Dataframe training set (train), apomonwnei 
#ta monadika stoixeia kai epistrefei pisw to set
def encode_data(df, train=None):
    #dn exw idea giati to kanei auto alla nvm
    df = df.copy()
    #epilegoume n asxolh8oume me tis sthles userId kai
    #movieId. gia ka8e mia apo tis 2 sthles epanelabe
    #thn diadikasia
    for col_name in ["userId", "movieId"]:
        #orizoume to train_col apo prin
        train_col = None
        #ka8ws to if mporei na mhn doulepsei
        #an yparxei training set epelekse thn ekastote
        #sthlh
        if train is not None:
            train_col = train[col_name]
        #dwse m pisw sto col mono thn deyterh timh
        #pou epistrefetai apo thn me8odo proc_col
        #(to array)
        #h proc_col opws eipame epistrefei pisw mono
        #monadika stoixeia
        _,col,_ = proc_col(df[col_name], train_col)
        #kane thn sthlh tou dataframe idia me thn sthlh
        #pou mas epistrafhke
        df[col_name] = col
        #epelekse apo thn sthlh tou dataframe mono ta
        #stoixeia pou einai megalytera h isa tou mhdenos
        df = df[df[col_name] >= 0]
    return df #dwse m pisw to epeksergasmeno dataframe

#as testaroume
#kalou kakou baze forward slashes sto path...
LOCAL_PATH = Path("J:/Shared folder/TEI/It has begun/Datasets/tiny")
#deikse mou to directory
print(list(LOCAL_PATH.iterdir()))
#fortwse ta dataset
#training
df_t = pd.read_csv(LOCAL_PATH/"tiny_training2.csv")
#validation
df_v = pd.read_csv(LOCAL_PATH/"tiny_val2.csv")
#deikse mou to training set
print(df_t)
#kwdikopoihse ta
df_t_e = encode_data(df_t) #training
df_v_e = encode_data(df_v, df_t) #validation
#deikse mou to kwdikopoihmeno training set
#parathrhse oti oi sthles userId kai movieId 
#exoun allaksei times kai exoun kwdikopoih8ei
#diaforetika!!
print(df_t_e)

#paradeigma enos embedding module
#ena embedding module einai enas aplos pinakas
#anazhthshs enos sta8erou dict kai twn diastasewn tou
#se auth thn periptwsh estw exoume 10 xrhstes me
#ena embedding size 3 (to mege8os tou embedding
#size einai o ari8mos twn paragontwn pou 8eloume
#na broume gia user/item)
#to embedding arxikopoieitai me tyxaies times
embed = nn.Embedding(10, 3)

#se ayth thn periptwsh anazhtoume thn timh gia ka8e
#ena apo ta id pou einai apo8hkeumena se enan
#tensor a
a = torch.LongTensor([[1,2,0,4,5,1]])
embed(a)

#montelo paragontopoihshs pinaka
class MF(nn.Module): #klhronomei to nn.Module
    def __init__(self, num_users, num_items, 
                 emb_size=100): #domhths
        super(MF, self).__init__()
        #users
        self.user_emb = nn.Embedding(num_users, emb_size)
        #items
        self.item_emb = nn.Embedding(num_items, emb_size)
        #kane ta barh se omoiomorfh katanomh
        #apo 0 ews 0.05
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)
        
    #gia id xrhsth u kai id item v, bres mou 
    #ta stoixeia tous kai epestrepse mou thn
    #thn problepomenh timh tou rating twn u gia
    #ta items v
    def forward(self, u, v):
        u = self.user_emb(u)
        v = self.item_emb(v)
        return (u*v).sum(1)
 
    
#as dokimasoume ena debugging session...
print('')
print(df_t_e)

num_users = 7
num_items = 4
emb_size = 3

#kanoume oti kaname pio panw
#ta embedding montela
user_emb = nn.Embedding(num_users, emb_size)
item_emb = nn.Embedding(num_items, emb_size)
#ta id
users = torch.LongTensor(df_t_e.userId.values)
items = torch.LongTensor(df_t_e.movieId.values)

#look up olous tous xrhstes kai ta items
U = user_emb(users)
V = item_emb(items)

print('')
print(U) #times paragontwn twn xrhstwn

print('')
print(U*V) #dot product

#deikse mou to a8roisma tou pollaplasiasmou twn
#paragontwn - to anamenomeno rating
print('')
print((U*V).sum(1))


#diadikasia ekapideushs
#afou exei ginei to encoding sto dataset
#dinoume thn timh tou se mia allh metablhth.
#dn einai anagkaio... alla gia xarin orismwn
#to df_train/df_test einai kalytero
df_train = df_t_e
df_test = df_v_e

#ari8mos xrhstwn kai antikeimenwn
num_users = len(df_train.userId.unique())
num_items = len(df_train.movieId.unique())
print('')
print(num_users, num_items)

#as ftiaksoume ena montelo basismeno sto MF module
#poy ftiaksame pio panw
model = MF(num_users, num_items, emb_size=100)

#kai as ftiaksoume mia methodo klhshs ekpaideushs
#ka8ws einai pio boliko na kaloume mia entolh
#ka8e fora para na grafoume ton algorithmo
#synexws!!
def train_epocs(model, epochs=10, lr=0.01, 
                wd=0.0, unsqueeze=False):
    #8ymasai ton Adam?
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=wd)
    #ksekinaei h ekpaideush
    for i in range(epochs):
        #user ids
        users = torch.LongTensor(df_train.userId.values)
        #item ids
        items = torch.LongTensor(df_train.movieId.values)
        #ratings
        ratings = torch.FloatTensor(df_train.rating.values)
        #unsqueeze? :/
        if unsqueeze:
            ratings = ratings.unsqueeze(1)
        #prediction model
        y_hat = model(users, items)
        loss = F.mse_loss(y_hat, ratings) #Loss
        #mhdenise gradients
        optimizer.zero_grad()
        loss.backward() #ypologise gradients
        optimizer.step() #update gradients
        print(loss.item()) #deikse mou to loss
    test_loss(model, unsqueeze)
    
#oriste ti kanei to test_loss parapanw
#methodos testing
def test_loss(model, unsqueeze=False):
    model.eval() #bazoume to montelo se evaluation mode
    #user kai item Ids kai ratings
    users = torch.LongTensor(df_test.userId.values) #.cuda()
    items = torch.LongTensor(df_test.movieId.values) #.cuda()
    ratings = torch.FloatTensor(df_test.rating.values)
    if unsqueeze: #unsqueeze?
        ratings = ratings.unsqueeze(1)
    y_hat = model(users, items) #prediction
    #kane test to loss- egine kala?
    loss = F.mse_loss(y_hat, ratings)
    print("test loss %.3f " % loss.item())
    
#test
#ta barh (paragontes) tou montelou paramenoun 
#enhmerwmena meta apo ka8e klhsh ths train_epocs, 
#me apotelesma me ka8e diadoxikh klhsh h apwleia 
#na meiwnetai
print('')
print('MF: ')
train_epocs(model, epochs=10, lr=0.1)
train_epocs(model, epochs=15, lr=0.01)
train_epocs(model, epochs=15, lr=0.006)

#matrix factorization with bias
class MF_bias(nn.Module):
    #constructor
    def __init__(self, num_users, num_items, emb_size=100):
        #call super constructor
        super(MF_bias, self).__init__()
        #user embedding, user bias
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        #item embedding, item bias
        self.item_emb = nn.Embedding(num_users, emb_size)
        self.item_bias = nn.Embedding(num_users, 1)
        #uniform(0, 0.05) weights
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)
        #uniform(-0.01, 0.01) biases
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)
    
    #forward function (prediction)
    def forward(self, u, v):
        #user, item weights
        U = self.user_emb(u)
        V = self.item_emb(v)
        #biases
        b_u = self.user_bias(u).squeeze()
        b_v = self.item_bias(v).squeeze()
        #make the prediction
        return (U*V).sum(1) + b_u + b_v
        
#hello? test
#ari8mos paragontwn = 100
model = MF_bias(num_users, num_items, emb_size=100)

print('')
print('MF_bias: ')
train_epocs(model, epochs=10, lr=0.018, wd=1e-5)
train_epocs(model, epochs=10, lr=0.01, wd=1e-5)
train_epocs(model, epochs=10, lr=0.001, wd=1e-5)

class CollabFNet(nn.Module):
    #n_hidden: ari8mos neurwnwn sto kryfo strwma
    def __init__(self, num_users, num_items,
                 emb_size=100, n_hidden=10):
        #kalese ton domhth ths anwterhs klashs (module)
        super(CollabFNet, self).__init__()
        #embeddings
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        #linear neural network module 1 
        #eisodos mege8ous emb_size*2 (8a deis giati meta),
        #eksodos mege8ous n_hidden
        self.lin1 = nn.Linear(emb_size*2, n_hidden)
        #linear neural network module 2
        #eisodos mege8ous n_hidden, eksodos mege8ous 1
        #dexetai thn eisodo ths lin1, bgazei 1 
        #output
        self.lin2 = nn.Linear(n_hidden, 1)
        #dropout function. to 0.1 ekfrazei pi8anothta
        #dropout 10% (p=0.1, p apo 0 ews 1). oi 
        #eisodoi akyrwnontai tyxaia symfwna me to 
        #p me mia katanomh Bernoulli kai ta 
        #stoixeia pou akyrwnontai,
        #to baros twn eksodwn tous apo to dropout
        #layer ginetai iso me 1/(1-p). akomh, o,ti 
        #dexetai san eisodo ena dropout layer bgazei
        #san eksodo, alliws kanonikopoiei thn eksodo
        #pollaplasiazontas me to baros
        self.drop1 = nn.Dropout(0.1)
       
    #twra giati kanei thn parakatw methodo san ta 
    #moutra tou den exw idea omws. to 8ema einai
    #oti bgazei ena output to opoio mporoume na
    #sygkrinoume me to rating kai gia ta opoia
    #modules p exoume ka8orisei (lin1, lin2)
    #mporoume na ypologisoume ta barh tous gia
    #na paroume mia ikanopoitikh apanthsh    
    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.item_emb(v)
        #torch.cat: concatenation kapoiwn tensors
        #(U kai V) sthn diastash pou orizetai (1)
        #relu: synarthsh rampas. opoiadhpote timh
        #katw apo mhden mhdenizetai.
        #to apotelesma einai ena concatenation
        #sthn prwth diastash panw sto opoio 
        #efarmozetai mia synarthsh rampas
        #(eksou kai to emb_size*2 input layer tou
        #lin1 pou dexetai eisodo idiwn diastasewn
        #me to output ths parakatw entolhs)
        x = F.relu(torch.cat([U, V], dim=1))
        #opoiodhpote stoixeio apo to parapanw tensor
        #exei 10% pi8anothta na ginei dropout me
        #ton tropo pou perigrafhke parapanw. poly
        #xrhsimo se periptwseis overfitting pou
        #8eloume na ksefortw8oume neurwnes gia
        #logous apofyghs ypermontelopoihshs
        x = self.drop1(x)
        #kane synarthsh rampas panw sto output tou
        #grammikou montelou lin1(x)
        x = F.relu(self.lin1(x))
        #ypologise thn telikh eksodo efarmozontas
        #to grammiko montelo lin2(x)
        x = self.lin2(x)
        return x #epistrofh
    
#ante na doume ti malakies exoume kanei me to
#parapanw gmtxm. btw dn yparxei pia matrix
#factorization. to CollabFNet einai kseka8ara
#ena sketo polystrwmatiko neural network so yeah...
model = CollabFNet(num_users, num_items, emb_size=100)

#opws eipame kai parapanw ta barh sto montelo
#paramenoun. opote ka8e klhsh tou train mas bgazei
#synexws kalytera apotelesmata.
#ka8ws omws ka8e fora pou ta neurwnika modules 
#(lin1, lin2) arxikopoiountai me tyxaia barh
#bgazoun kai diaforetika apotelesmata, oxi mono se
#diaforetika treksimata alla kai metaksy montelwn
#pou arxikopoihsame se ena run (MF, MF_bias,
#CollabFNet)
print('')
print('CollabFNet: ')
train_epocs(model, epochs=15, lr=0.05, wd=1e-6, unsqueeze=True)
train_epocs(model, epochs=10, lr=0.01, wd=1e-6, unsqueeze=True)
train_epocs(model, epochs=10, lr=0.001, wd=1e-6, unsqueeze=True)
train_epocs(model, epochs=10, lr=0.001, wd=1e-6, unsqueeze=True)

#opws 8a katalabeis an diabaseis kai to neighbourhood-
#based arxeio, oi neighbourhood-based kai oi learning-
#based methodoi den exoun *KAMIA* apolytws sxesh kai
#egw thn exw gamhsei prospa8wntas n ma8w kai apo tis
#dyo alla kai n dokimasw parallhlopoihsh learning-
#based methodwn opote ton hpia gia ta kala