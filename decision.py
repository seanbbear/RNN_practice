import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
import numpy

from gensim.models import word2vec
from gensim import models
import logging

import numpy
import jieba
from udicOpenData.stopwords import *

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return F.sigmoid(x)

trained = torch.load("trained.pkl")
model = models.Word2Vec.load('word2vec+reply.model')

while True:
    a = input("請輸入一句話: ")
    count = []
    seg = list(rmsw(a))
    print(seg)
    for i in range(len(seg)):
    	if seg[i] in model.wv.vocab:
    		count.append(model[seg[i]])

    if count:
        mean_vec=numpy.mean(count, axis=0)##各pos向量之平均
    else:
        mean_vec=numpy.zeros(200)
    x = torch.tensor(mean_vec,dtype = torch.float)
    p_rate = trained(x).tolist()[0]
    n_rate = trained(x).tolist()[1]
    if p_rate>n_rate:
        print("正面",p_rate,n_rate)
    else:
        print("負面",p_rate,n_rate)


