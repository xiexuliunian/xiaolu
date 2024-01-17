from torch.utils.data import Dataset
import torch
import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb

train_data_path = "/work/xiaolu/uplod/data1127.csv"
test_data_path="/work/xiaolu/uplod/data226.csv"

def precoss(a):
    a[0] = a[0] - 20
    a[2] = a[2] + 20
    a[3] = a[3] - 20
    a[5] = a[5] + 20
    a[6] = a[6] - 20
    a[8] = a[8] + 20

    a[9] = a[9] + 20
    a[10] = a[10] + 20
    a[11] = a[11] + 20
    a[15] = a[15] - 20
    a[16] = a[16] - 20
    a[17] = a[17] - 20
    b = [a[0], a[9], a[1], a[10], a[2], a[11], a[3], a[12], a[4], a[13], a[5], a[14], a[6], a[15], a[7], a[16], a[8],
         a[17]]
    return b

def countf(alist):
    number = alist[0] * 32 + alist[1] * 16 + alist[2] * 8 + alist[3] * 4 + alist[4] * 2 + alist[5] * 1
    return number

def acc(filepath):
    oridata = pd.read_csv(filepath)
    right=0
    for i in range(oridata.shape[0]):
        a = [oridata.iloc[i, 0], oridata.iloc[i, 1], oridata.iloc[i, 2], oridata.iloc[i, 3], oridata.iloc[i, 4],
             oridata.iloc[i, 5], oridata.iloc[i, 6], oridata.iloc[i, 7], oridata.iloc[i, 8],
             oridata.iloc[i, 9], oridata.iloc[i, 10], oridata.iloc[i, 11], oridata.iloc[i, 12], oridata.iloc[i, 13],
             oridata.iloc[i, 14], oridata.iloc[i, 15], oridata.iloc[i, 16], oridata.iloc[i, 17]]
        b = precoss(a)
        b_in = torch.FloatTensor(b).cuda()
        out = net(b_in)
        out = sigmoid(out)
        ll = out.tolist()
        for j in range(6):
            if ll[j] > 0.5:
                ll[j] = 1
            else:
                ll[j] = 0
        gt=countf([oridata.iloc[i, 18], oridata.iloc[i, 19], oridata.iloc[i, 20], oridata.iloc[i, 21], oridata.iloc[i, 22],
             oridata.iloc[i, 23]])
        predict=countf(ll)
        if predict==gt:
            right +=1

    return right/oridata.shape[0]


wandb.init(project="transformer")
class MyData(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.data.iloc[:, 0] = self.data.iloc[:, 0] - 20
        self.data.iloc[:, 2] = self.data.iloc[:, 2] + 20
        self.data.iloc[:, 3] = self.data.iloc[:, 3] - 20
        self.data.iloc[:, 5] = self.data.iloc[:, 5] + 20
        self.data.iloc[:, 6] = self.data.iloc[:, 6] - 20
        self.data.iloc[:, 8] = self.data.iloc[:, 8] + 20

        self.data.iloc[:, 9] = self.data.iloc[:, 9] + 20
        self.data.iloc[:, 10] = self.data.iloc[:, 10] + 20
        self.data.iloc[:, 11] = self.data.iloc[:, 11] + 20
        self.data.iloc[:, 15] = self.data.iloc[:, 15] - 20
        self.data.iloc[:, 16] = self.data.iloc[:, 16] - 20
        self.data.iloc[:, 17] = self.data.iloc[:, 17] - 20



        self.data_len = len(self.data)

    def __getitem__(self, index):
        tmp = self.data.iloc[index, :18].astype(float)
        data = torch.FloatTensor([[tmp[0], tmp[9]],
                                  [tmp[1], tmp[10]],
                                  [tmp[2], tmp[11]],
                                  [tmp[3], tmp[12]],
                                  [tmp[4], tmp[13]],
                                  [tmp[5], tmp[14]],
                                  [tmp[6], tmp[15]],
                                  [tmp[7], tmp[16]],
                                  [tmp[8], tmp[17]],
                                  ]).reshape(18)
        label = torch.FloatTensor(self.data.iloc[index, 18:24].astype(int))
        return data, label

    def __len__(self):
        return self.data_len


# 从iris.csv文件建立训练数据集
train_dataset=MyData(train_data_path)
test_dataset = MyData(test_data_path)
# 产生数据生成器，每次产生32个数据
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)


#transformer 网络结构


net = Net(18, 6).cuda()
wandb.watch(net,log_freq=100)
net.train()
# 使用SGD一阶函数优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
# 如果分类类别是互斥的应该使用普通交叉熵损失函数，但考虑到患者可能是多条肌肉受损，
# 也就是可能出现标签为【1,1,1】的情况，则应该使用二元交叉熵函数，每个类别独立计算可能性
loss_func = torch.nn.BCEWithLogitsLoss()
sigmoid = torch.nn.Sigmoid()

num_epochs=1000

for epoch in (tqdm(range(num_epochs))):
    num_trainloss=0
    num_testloss = 0
    for i, (data, label) in enumerate(test_dataloader):
        out = net(data.cuda())
        # print(out.shape)
        label = label.squeeze(1)
        loss = loss_func(out, label.cuda())
        num_testloss+=loss.item()
    test_acc=acc(test_data_path)
    # print("test_acc",test_acc)
    wandb.log({"test_acc": test_acc})
    wandb.log({"test_loss": num_testloss},commit=False)

    for i, (data, label) in enumerate(train_dataloader):
        out = net(data.cuda())
        # print(out.shape)
        label = label.squeeze(1)
        loss = loss_func(out, label.cuda())
        num_trainloss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_acc = acc(train_data_path)
    # print("train_acc",train_acc)
    wandb.log({"train_acc": train_acc},commit=False)
    wandb.log({"train_loss": num_trainloss},commit=False)
    

torch.save(net, "modeltransformer-wandb.pth")
