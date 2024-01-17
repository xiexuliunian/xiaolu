from torch.utils.data import Dataset
import torch
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm

data_path = "/work/xiaolu/uplod/data1127.csv"
print(data_path)

data = pd.read_csv(data_path)


class MyData(Dataset):
    def __init__(self, csv_path):
        self.data = data
        self.data.iloc[:, 0] = data.iloc[:, 0] - 20
        self.data.iloc[:, 2] = data.iloc[:, 2] + 20
        self.data.iloc[:, 3] = data.iloc[:, 3] - 20
        self.data.iloc[:, 5] = data.iloc[:, 5] + 20
        self.data.iloc[:, 6] = data.iloc[:, 6] - 20
        self.data.iloc[:, 8] = data.iloc[:, 8] + 20

        self.data.iloc[:, 9] = data.iloc[:, 9] + 20
        self.data.iloc[:, 10] = data.iloc[:, 10] + 20
        self.data.iloc[:, 11] = data.iloc[:, 11] + 20
        self.data.iloc[:, 15] = data.iloc[:, 15] - 20
        self.data.iloc[:, 16] = data.iloc[:, 16] - 20
        self.data.iloc[:, 17] = data.iloc[:, 17] - 20

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
xl_dataset = MyData(data_path)
# 产生数据生成器，每次产生32个数据
dataloader = torch.utils.data.DataLoader(dataset=xl_dataset, batch_size=128, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super(Net, self).__init__()
        # 将输入的4维映射到100维高维空间
        self.hidder1 = torch.nn.Linear(in_feature, 256)
        self.hidder2 = torch.nn.Linear(256, 512)
        self.hidder3 = torch.nn.Linear(512, 256)
        self.out = torch.nn.Linear(256, out_feature)

    def forward(self, x):
        x = F.relu(self.hidder1(x))
        x = F.relu(self.hidder2(x))
        x = F.relu(self.hidder3(x))
        x = self.out(x)
        return x


# 初始化网络为输入4维，输出3维的网络结构
net = Net(18, 6).cuda()
net.train()
# 使用SGD一阶函数优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
# 如果分类类别是互斥的应该使用普通交叉熵损失函数，但考虑到患者可能是多条肌肉受损，
# 也就是可能出现标签为【1,1,1】的情况，则应该使用二元交叉熵函数，每个类别独立计算可能性
loss_func = torch.nn.BCEWithLogitsLoss()

for num in (tqdm(range(500))):
    for i, (data, label) in enumerate(dataloader):
        out = net(data.cuda())
        label = label.squeeze(1)
        loss = loss_func(out, label.cuda())
        print("loss:", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(net, "modeldeeper.pth")
