import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
import numpy

pos_mean_vec = numpy.loadtxt('PosMeanVec.txt')
neg_mean_vec = numpy.loadtxt('NegMeanVec.txt')
torch.tensor(pos_mean_vec,dtype = torch.float)
print(pos_mean_vec)

BATCH_SIZE = 5
torch_dataset = Data.TensorDataset(pos_mean_vec,neg_mean_vec)
loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

for epoch in range(10):
    for step,(x, data) in enumerate(loader):#enumerate:增加索引
        # print(x)
        out = net(x)
        loss = loss_func(out,data)
        # input(out.size())
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()