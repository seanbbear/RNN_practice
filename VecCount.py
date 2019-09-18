from gensim.models import word2vec
from gensim import models
import logging

import numpy
import jieba
from udicOpenData.stopwords import *
pos_mean_vec=[]
neg_mean_vec=[]
testdata=[]
def vec_count():
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = models.Word2Vec.load('word2vec+reply.model')######################
    
    pos_rply = []
    neg_rply = []
    with open('3000_tw.txt','r',encoding='utf-8') as f:
        word = f.readlines()
        for i in range(len(word)):
            if "1" in word[i][0]:
                pos_rply.append(word[i])
            elif "0" in word[i][0]:
                neg_rply.append(word[i])

    
    
    
    for i in range(len(pos_rply)):
        a=list(rmsw(pos_rply[i]))
        for j in range(len(a)):
            count=[]
            if a[j] in model.wv.vocab:
                count.append(model[a[j]])
        if count:
            pos_mean_vec.append(numpy.mean(count, axis=0))##各pos向量之平均
        else:
            pos_mean_vec.append(numpy.zeros(200))
    
    for i in range(len(neg_rply)):
        b=list(rmsw(neg_rply[i]))

        for j in range(len(b)):
            count1=[]
            if b[j] in model.wv.vocab:
                count1.append(model[b[j]])
        if count1: 
            neg_mean_vec.append(numpy.mean(count1, axis=0))##各neg向量之平均
        else:
            neg_mean_vec.append(numpy.zeros(200))

    # print(len(pos_mean_vec))
    # print(len(neg_mean_vec))
    numpy.savetxt('PosMeanVec.txt',pos_mean_vec)
    numpy.savetxt('NegMeanVec.txt',neg_mean_vec)
    # with open('PosMeanVec.txt','w',encoding='utf-8') as f: 
    #     for i in pos_mean_vec:
    #         f.write(i)
    #         f.write('\n')
    # with open('NegMeanVec.txt','w',encoding='utf-8') as f: 
    #     for i in neg_mean_vec:
    #         f.write(i)
    #         f.write('\n')
vec_count()