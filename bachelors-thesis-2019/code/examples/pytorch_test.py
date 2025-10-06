# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 02:13:21 2018

@author: Ceyx
"""
#imports
import torch #torch
import torch.autograd as autograd #autograd
import torch.nn as nn #nn module (neural network)
import torch.nn.functional as F #synarthseis
import torch.optim as optim #optimizer (gradient update)
import numpy as np #numpy
import matplotlib.pyplot as plt #diagrammata

#tensor

#tropos 1
N=5 #diastash row
x=torch.randn(N,10).type(torch.FloatTensor)
#tropos 2
dtype = torch.float #eidos metablhths
device = torch.device("cpu") #syskeuh - cpu/gpu
x=torch.randn(N, 10, device=device, dtype=dtype)

print(f'\nx = {x}') #kai oi 2 tropoi bgazoun tensor

#tensor reshaping
y = x.view(1,-1) #to -1 shmainei kane ayth th diastash oso prepei
#symfwna me tis ypoloipes diastaseis p s dinw sto view

print(f'\ny = {y}')

#autograd - dhmiourgei aytomata gradients se tensors
#to parakatw dhmiourgei enan float tensor me gradient
x = torch.tensor([1.,2.,3.,4.,5.,6.], requires_grad=True)
print(f'\nGradient of x is {x.grad}') #den yparxei akomh gradient!!

L = (2*x+1).sum() #pollaplasiazontai ola ta stoixeia tou pinaka x
#epi 2 kai prosti8etai se ayta 1
#ystera brisketai to synolo olwn twn stoixeiwn tou pinaka x

#ypologizetai to gradient olwn twn tensors pou exoun gradient
#symfwna me to L
L.backward()

print('\nGradient of x is not None anymore!! Check again!!')
print(x.grad)

#to torch.nn module (neural network)
#parakatw ftiaxnoume ena linear module
D=5 #input features
M=3 #neurons - output
linear_map = nn.Linear(D,M)
#oi parametroi sto grammiko montelo arxikopoiountai tyxaia
#parathrhse oti yparxei o parakatw pinakas
#me ta barh (parameters)
#to prwto set einai enas pinakas a 
#enw to deutero enas pinakas apo b (pou 8a doume parakatw)
#(a*x+b) 
print(f'\nLinear model parameters: {[p for p in linear_map.parameters()]}')

#Fake data
def lin(a,b,x): return a*x+b #mia methodos grammikou montelou

#methodos dhmiourgias pseytikwn dedomenwn
def gen_fake_data(n, a, b):
    #epistrefei enan pinaka apo n stoixeia apo 0 ews 1
    #omoiomorfa (uniform)
    #epistrefei ena numpy.ndarray h scalar
    #ndarray an dwseis mege8os, scalar (monos ari8mos)
    #an den oriseis
    #auto einai oi arxikes times
    x = np.random.uniform(0, 1, n)
    #epistrefei ena ndarray
    #pros8etontas thn timh mias grammikhs synarthshs
    #sto x kai to pollaplasiasmo enos ndarray 1000 stoixeiwn
    #me 0.1
    #auto einai oi pragmatikes times 
    #(me mia mikrh timh 8orybou)
    y = lin(a,b,x)+0.1*np.random.normal(0,3,n)
    return x,y #epistorfh twn array se ena tuple

#50 samples, a=3(float),b=8(float)
x,y = gen_fake_data(50,3.,8.)

#draw plots
#s - scalar of array_like (x, y)
#xlabel - etiketa tou x
#ylabel - etiketa tou y
plt.scatter(x, y, s=8); plt.xlabel("x"); plt.ylabel("y");
plt.show()

#mean squared error
#dexetai 2 ari8mous (h kai array ari8mwn), y
#kai y_hat (ena kapelo tou y) kai sygkrinei 
#metaksy tous to mesο tetragwniko sfalma
#to y_hat einai to shmeio anaforas
#pros sygkrish me to y
def mse(y_hat, y): return ((y_hat - y)**2).mean()

#as ypo8esoume oti to y_hat se ayth th periptwsh einai
#mia grammikh synarthsh me a=10 kai b=5
#parathrhsh: to x kai to y ta exoume dhmiourghsei hdh
#pio panw (gen_fake_data)
y_hat = lin(10,5,x)
#ypologizoume to mse anamesa se y_hat kai y
print(f'\nMean squared error (y, y_hat): {mse(y_hat, y)}')

#mse loss
#dhmiourgoume mia methodo opou mporoume na epistrepsoume
#to mse anamesa se mia grammikh synarthsh kai ena y
def mse_loss(a,b,x,y): return mse(lin(a,b,x),y)
print(f'\nMean squared error (10*x+5, y): {mse_loss(10, 5, x, y)}')

#as kanoume ligo gradient descent
#kainouria fake data
#10000 samples, a=3(float), b=8(float)
x, y = gen_fake_data (10000, 3., 8.)
#pairnoume pisw tis diastaseis
#se auth th periptwsh kai ta 2 exoun mia diastash
#10000 keliwn
print(f'\nx.shape: {x.shape} y.shape: {y.shape}')

#as kanoume ta x kai y pytorch tensors!!!
#prosoxh sto oti to kai y den exoun gradients
#mono ta barh dexontai gradients ka8ws auta einai
#pou prokeitai na allazoun
x = torch.tensor(x)
y = torch.tensor(y)

#kai as ftiaxoume tyxaia tensor barh a kai b
#me gradients poy mporoun na ypologistoun kata thn
#backward fash
#h randn(1) epistrefei enan pinaka me 1 stoixeio
#se mia diastash
a, b = np.random.randn(1), np.random.randn(1)
#casting se tensors...
a = torch.tensor(a, requires_grad=True)
b = torch.tensor(b, requires_grad=True)
#deikse mou ti eftiakses
print(f'\na= {a}, b= {b}\n')

#exontas ta barh mas a, b, to x kai to y as ftiaksoume loipon
#twra ena systhma pou 8a beltistopoiei ta barh
#lynontas ws pros thn elaxistopoihsh tou mse
#metaksy mias grammikhs synarthshs a*x+b kai y

learning_rate = 1e-3 #taxythta sygklishs
#gia 10000 epoxes kane mou ta parakatw
for t in range(10000):
    #gia ta barh a kai b ypologise mou to
    #meso tetragwniko sfalma 
    loss = mse_loss(a,b,x,y)
    #ana 1000 bhmata deikse mou to mse
    #to item() deixnei thn timh tou tensor mono
    #kai oxi oles tis plhrofories
    if t%1000 == 0: print(loss.item())
    
    #symfwna me to loss ypologise mou se ola
    #ta tensors pou exoun, to gradient
    loss.backward()
    
    #kane mou update ta barh symfwna me thn taxythta
    #sygklishs kai ta barh
    a.data -= learning_rate * a.grad.data
    b.data -= learning_rate * b.grad.data
    
    #kane ta gradients zero gia thn epomenh epanalhpsh
    a.grad.data.zero_()
    b.grad.data.zero_()
    
#poia einai ta barh loipon twra?
print(f'\na = {a.item()},\nb = {b.item()}')


#enas poly aplos tropos n kanoume to idio montelo
#syntoma kai grhgora einai o ekshs
#to Sequential einai ena container neurwnikwn
#diktywn pou ginontai sth seira
model = nn.Sequential(
        #se ayth thn periptwsh orizoume ena grammiko
        #montelo me mia eksodo kai mia eisodo
        #praktika mia aplh grammikh synarthsh
        nn.Linear(1,1)
        )

#enas diaforetikos (epekshghmatikos) tropos einai o parakatw
class LinearRegression(nn.Module): #klhronomei thn klash Moduke
    def __init__(self): #class initialization
        #kalei ton domhhth ths klasshs Module
        super(LinearRegression, self)._init_()
        #orizetai metablhth lin opou dinetai h timh enos
        #grammikou neurwnikou montelou me 1 eksodo kai 1
        #eisodo
        self.lin = nn.Linear(1,1)
        
    def forward(self,x): #eksodos montelou
        #eisagontas ena x ypologizoume thn timh tou
        #symfwna me to grammiko montelo lin
        x = self.lin(x)
        return x #epistrofh tou x
    
#as doume tis parametrous tou model
print('')#kenh grammh
print([p for p in model.parameters()])

#as ftiaksoume akoma mia fora pseutika dedomena
x, y = gen_fake_data(10000, 3., 8.)
#kai as ta kanoume float tensors
x = torch.tensor(x).float()
y = torch.tensor(y).float()
#poio einai to sxhma twn diastasewn tou x?
print(f'\nx.shape: {x.shape}')

#prosoxh stis diastaseis pou to montelo sou perimenei!!
#me thn leitourgia unsqueeze mporoume na dhmiourghsoume
#ena tensor apo ena allo kai na pros8esoume diastaseis
#se auth thn periptwsh pros8etoume mia diastash 1 keliou
#sto tensor x
x1 = torch.unsqueeze(x,1)
print(f'\nx1.shape: {x1.shape}')

#dhmiourgoume to y_hat(prediction) symfwna me
#to x1 (arxikes times)
#einai anagkaio na ginei katanohto oti ta barh yparxoun
#mesa sto idio to *model* ka8ws einai to grammiko montelo mas
#(8ymisou oti to grammiko montelo apotelei mia synarthsh
#y=a*x+b opou a kai b ta barh)
y_hat = model(x1)
print('')
print(y_hat)

#pame loipon na ypologisoume ta barh wste na proseggisoume
#thn pragmatikh synarthsh (me 8orybo)
learning_rate = 0.1 #taxythta sygklishs
#8a xrhsimopoihsoume enan optimizer pou 8a enhmerwsei ta
#barh tou montelou gia emas, se ayth thn periptwsh ton Adam
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#10.000 epoxes
for t in range(10000):
    #ypologise to y_hat (prediction)
    y_hat = model(x1)
    #ypologise to meso tetragwniko sfalma anamesa se
    #y kai y_hat
    #parathrhse oti gia prwth fora xrhsimopoioume
    #methodo tou torch.nn.functional anti gia thn dikh mas
    loss = F.mse_loss(y_hat, y.unsqueeze(1))
    #ana 1000 epoxes emfanise mou thn timh tou mse
    if t % 1000 == 0: print(loss.item())
    
    #exe sto nou sou oti ka8e fora prepei na mhdenistoun ta
    #gradients!!
    #edw xrhsimopoioume ton optimizer gia na to kanei auto
    optimizer.zero_grad()
    #dwse mou ta gradients symfwna me to loss
    loss.backward()
    
    #ananewse ta barh symfwna me ta gradients
    optimizer.step()
    
#afou teleiwses twra loipon dwse mou ta telika barh tou model
#as exoyme ypopsin oti to a kai b ksekinhsan ws 3 kai 8
#opote opoiadhpote timh ta proseggizei einai mia kaly lysh!! 
print('')#kenh grammh
print([p for p in model.parameters()])

#logistikh palindromhsh
#as ftiaksoume mia diaforetikh ekdosh tou gen_fake_data
#me logistiko anti gia grammiko montelo

#grammikh synarthsh
def lin(a,b,x): return a*x+b

#n diastaseis, a,b barh
def gen_logistic_fake_data(n, a, b):
    #omoiomorfh katanomh anamesa se -20 kai 20
    #me 2 diastaseis, n (1h) kai 2(2h)
    x = np.random.uniform(-20, 20, (n, 2))
    #ftiaxnoume ena grammiko threshold (hat) xrhsimopoiwntas
    #ta prwta kelia ths deyterhs diastashs
    x2_hat = lin(a, b, x[:,0])
    #sygkrinoume ta deutera kelia ths deuterhs diastashs
    #me to hat kai apo8hkeuoume tis times true an
    #einai megalyterou mege8ous kai false an einai
    #mikroterou se ena array diastasewn (n)
    y = x[:,1] > x2_hat
    #prosoxh! to y prepei na epistrafei san ari8mos ka8ws
    #einai boolean!
    return x, y.astype(int)

#n=100, a=1(float), b=0.5(float)
x, y = gen_logistic_fake_data(100, 1., 0.5)
print(f'\ny = {y}') #gia na doume...

#ena euros ari8mwn apo -20 ews 20 me bhma 0.2
#h diaxwristikh mas grammh!
t = np.arange(-20, 20, 0.2)
#skorpizoume to x san dyo diastaseis. thn timh tou x 8a
#paroun ta prwta kelia ths deuterhs diastashs (x[:,0])
#enw thn timh tou y ta deutera (x[:,1])
#to xrwma twn stoixeiwn panw apo thn diaxwristikh grammh
#8a einai kitrino (y-yellow) kai o scalar 8a einai 8
plt.scatter(x[:,0], x[:,1], c=y, s=8)
plt.xlabel('x1'); plt.ylabel('x2');
#mesa sto grafhma bale mou kai thn diaxwristh grammh
#thn opoia 8a emfaniseis me kokkines diakekommenes
#grammes
plt.plot(t, t + 0.5, 'r--')
plt.show()#deikse to grafhma

#x, y arrays to tensors
x = torch.tensor(x).float()
y = torch.tensor(y).float()

#grammiko montelo, 2 input, 1 output
model = torch.nn.Sequential(
        torch.nn.Linear(2,1)
        )
print(f'\nModel: {model}')#ti ftiaksame??
print(f'\nModel dimensions: {model(x).shape}')#diastaseis

#okay... as ftiaksoume ena megalytero data set
#n = 10.000, a=1.(float), b=0.5(float)
x, y = gen_logistic_fake_data(10000, 1., 0.5)
#se tensors
x = torch.tensor(x).float()
y = torch.tensor(y).float()

#diadikasia ma8hshs
learning_rate = 0.1 #taxythta sygklishs
#dinoume akomh enan optimizer pou 8a kanei olh thn douleia
#gia emas. epilegoume ksana ton Adam
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#10.000 epoxes
for t in range(10000):
    y_hat = model(x)#prediction
    #se ayth thn periptwsh xrhsimopoioume binary cross
    #entropy gia ton ypologismo ths apwleias
    #(to opoio den exw idea ti einai so please mhn me
    #rwtas, htan aplws mesa sto tutorial gmtxm)
    #anamesa se mia sigmoeidh synarthsh ws to y_hat
    #kai to y (kanoume unsqueeze gia logous symbatothtas!!)
    loss = F.binary_cross_entropy(torch.sigmoid(y_hat), y.unsqueeze(1))
    #ana 1000 epoxes deikse mou thn apwleia
    if t % 1000 == 0: print(loss.item())
    #ka8e fora mhdenizoume ta gradients!! must!!!
    optimizer.zero_grad()
    #dwse mou pisw ta gradients analoga me to loss
    loss.backward()
    #kai ananewse ta barh
    optimizer.step()
    
#deikse ta barh
print ([p for p in model.parameters()])

#Data loaders kai SGD!!
#sto SGD (stochastic gradient descent) h diadikasia einai
#idia me to aplo gradient descent me thn diafora oti
#anti gia thn xrhsh olwn twn deigmatwn se ka8e treksimo
#xrhsimopoioume ena mono subset twn deigmatwn
#gnwsto ws batch h minibatch

#as ftiaksoume ena grammiko montelo
#me mia eisodo kai mia eksodo
model2 = torch.nn.Sequential(
        torch.nn.Linear(1,1)
        )

#xreiazomai tis biblio8hkes Dataset kai DataLoader
#gia ton skopo pou ekshgh8hke parapanw!!
from torch.utils.data import Dataset, DataLoader

#ftiaxnoume pseutika dedomena
def lin(a,b,x): return a*x+b #grammikh synarthsh

def gen_fake_data(n, a, b):
    #omoiomorfh katanomh n deigmatwn apo 0 ews 1
    x = np.random.uniform(0, 1, n)
    #grammikh synarthsh me 8orybo
    y = lin(a, b, x) + 0.1*np.random.normal(0,3,n)
    #epistrofh x kai y san float
    return x.astype(np.float32), y.astype(np.float32)

#as ftiaksoume ena Dataset Palindromhshs!!
class RegressionDataset(Dataset): #klhronomei to Dataset
    #dexontai eisodoi a, b kai n me default times
    #3, 8 kai 10000
    def __init__(self, a=3, b=8, n=10000):
        #ftiakse mou pseutika dedomena gia na doulepsw
        x, y = gen_fake_data(n, a, b)
        #tensors!!
        x = torch.from_numpy(x).unsqueeze(1)
        y = torch.from_numpy(y)
        #orizw metablhtes klashs
        self.x, self.y = x, y
    
    #mege8os dataset
    def __len__(self):
        return len(self.y)
    
    #8elw pisw ta antikeimena index idx apo to x kai to y
    #epistrofh san tupples!
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
#ftiakse mou ena dataset me pseutika dedomena
fake_dataset = RegressionDataset()
#ftiaxnoume enan DataLoader panw sto fake dataset
#me to mege8os ka8enos apo ta batch na einai 1000
#enw einai dynato na anakateuoume ta dedomena gia ka8e
#batch (shuffle)
dataloader = DataLoader(fake_dataset, batch_size=1000, shuffle=True)
#ena paradeigma *trabhgmatos* dedomenwn apo ton loader
x, y = next(iter(dataloader))
#as doume ti periexei to y!!
#(casting san float tensor)
print(f'\ny: {y.type(torch.FloatTensor)}')

#diadikasia ma8hshs: 
#sygklish, optimizer, epoxes, loss, gradient update
#weight update
learning_rate = 0.1
optimizer = optim.Adam(model2.parameters(), lr=learning_rate)

for t in range(1000):
    #gia ka8e ena apo ta batch ypologizoume epanhlhmmena
    #10 batch - pio argo sthn praksh!!
    for i, (x,y) in enumerate(dataloader):
        y_hat = model2(x)
        loss = F.mse_loss(y_hat, y.unsqueeze(1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if t%100==0: print(loss.item())
    
print([p for p in model2.parameters()])

#as dokimasoume me ena neurwniko diktyo 2 strwmatwn!!

#fake data
def sigmoid(x): #sigmoeidhs synarthsh
    return 1/(1 + np.exp(-x))

def gen_nn_fake_data(n):
    #omoiomorfh katanomh diastasewn (n,2)
    x = np.random.uniform(0, 10, (n,2))
    x1 = x[:,0] #prwta kelia 2hs diastashs
    x2 = x[:,1] #deytera kelia
    #estw mia sigmoeidhs (neurwnas prwtou strwmatos)
    score1 = sigmoid(-x1 -8* x2 +50)
    #kai mia akoma (neurwnas prwtou strwmatos)
    score2 = sigmoid(-7*x1 - 2* x2 +50)
    #synarthsh poy xrhsimopoiei tis sigmoeideis
    #(neurwnas deuterou strwmatos)
    score3 = 2* score1 + 3*score2 - 0.1
    #poia items ths score 3 einai mikrotera tou 0?
    #(boolean)
    y = score3 < 0
    #casting apo boolean se int (0 h 1)
    return x, y.astype(int)

#fake data
x, y = gen_nn_fake_data(500)

#kane mou to plot
plt.scatter(x[:,0], x[:,1], c=y, s=8)
plt.xlabel('x1'); plt.ylabel('x2')
plt.show()

#as ftiaksoume to neurwniko mas diktyo
#grammikh-sigmoeidhs-grammikh
model = torch.nn.Sequential(
        torch.nn.Linear(2,2),
        torch.nn.Sigmoid(),
        torch.nn.Linear(2,1)
        )

print(f'\nTwo layer neural network model: {model}\n')

#ftiakse ta fake data, balta se float tensors
x, y = gen_nn_fake_data(10000)
x = torch.tensor(x).float()
y = torch.tensor(y).float()

#diadikasia ma8hshs
learning_rate = 0.01 #spicy!! mikrh timh!
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for t in range (10000): #10.000 epoxes
    y_hat = model(x)
    loss = F.binary_cross_entropy(torch.sigmoid(y_hat), y.unsqueeze(1))
    if t%1000 == 0: print(loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
#barh?
print ('\nTwo layer neural networks weights: ')
print ([p for p in model.parameters()])

#GJ bro. Not too shaby!