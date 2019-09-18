import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
import numpy

# from gensim.models import word2vec
# from gensim import models
# import logging

# import jieba
# from udicOpenData.stopwords import *






pos_mean_vec = numpy.loadtxt('PosMeanVec.txt')
neg_mean_vec = numpy.loadtxt('NegMeanVec.txt')




class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden,n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):   
        x = F.relu(self.hidden(x))# activation function for hidden layer
        x = self.out(x)
        return x
        #return F.torch.sigmoid(x)



net = Net(n_feature=200, n_hidden=25,n_output=2)     # define the network
# print(net)  # net architecture
lrate=0.0001
# optimizer = torch.optim.SGD(net.parameters(),lr=lrate)
optimizer = torch.optim.Adam(net.parameters(),lr=lrate)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()

# data產生
ze=[0]*3000
on=[1]*3000
da=ze+on
data = torch.tensor(da,dtype = torch.long)

# onedata = torch.ones(3000,dtype = torch.long)
# zerodata = torch.zeros(3000,dtype = torch.long)
# data = torch.cat((onedata, zerodata), 0).type(torch.LongTensor) 
# print(onedata)
mean_vec = list(pos_mean_vec) + list(neg_mean_vec)

# mean_vec = torch.cat((pos_mean_vec, neg_mean_vec), 1).type(torch.LongTensor) 
x = torch.tensor(mean_vec,dtype = torch.float)

# print(x.size())
# ----------------------------------
BATCH_SIZE = 10
torch_dataset = Data.TensorDataset(x,data)
loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=1,
)
# ----------------------------------
for epoch in range(50):
    lossVal = 0.0
    for step,(x, data) in enumerate(loader):#enumerate:增加索引
        optimizer.zero_grad()
        out = net(x)
        loss = loss_func(out,data)
        lossVal = lossVal + loss.item()
        # input(out.size())
        loss.backward()
        optimizer.step()
    lrate *= 0.8 #動態調整learning rate
    print(lossVal/step)
    # print(loss)
    

torch.save(net,"trained.pkl") 


poss1=0
negg1=0
trained=torch.load("trained.pkl")
for i in range(3000):
    x = torch.tensor(neg_mean_vec[i],dtype = torch.float)
# print(x)
    pos=trained(x).tolist()[0]
    neg=trained(x).tolist()[1]
    if pos>neg:
        poss1+=1
    else:
        negg1+=1

poss=0
negg=0
for i in range(3000):
    x = torch.tensor(pos_mean_vec[i],dtype = torch.float)
# print(x)
    pos=trained(x).tolist()[0]
    neg=trained(x).tolist()[1]
    if pos>neg:
        poss+=1
    else:
        negg+=1
print("正確率:",(negg1+poss)/6000)
print("答案為正面之正確率:",poss/3000)
print("答案為負面之正確率:",negg1/3000)




# print(variable)

