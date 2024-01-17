import torch
import torch.nn.functional as F
import pandas as pd


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


net = torch.load("/work/xiaolu/trainBCE/modeldeeper.pth").cpu()
print(net)
net.eval()

sigmoid = torch.nn.Sigmoid()


def countf(alist):
    number = alist[0] * 32 + alist[1] * 16 + alist[2] * 8 + alist[3] * 4 + alist[4] * 2 + alist[5] * 1
    return number


if __name__ == "__main__":
    oridata = pd.read_csv("/work/xiaolu/uplod/data226.csv")
    print("数据维度:",oridata.shape)
    right = 0
    for i in range(oridata.shape[0]):
        a = [oridata.iloc[i, 0], oridata.iloc[i, 1], oridata.iloc[i, 2], oridata.iloc[i, 3], oridata.iloc[i, 4],
             oridata.iloc[i, 5], oridata.iloc[i, 6], oridata.iloc[i, 7], oridata.iloc[i, 8],
             oridata.iloc[i, 9], oridata.iloc[i, 10], oridata.iloc[i, 11], oridata.iloc[i, 12], oridata.iloc[i, 13],
             oridata.iloc[i, 14], oridata.iloc[i, 15], oridata.iloc[i, 16], oridata.iloc[i, 17]]
        b = precoss(a)
        b_in = torch.FloatTensor(b)
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


    print("正确数目：",right)
    print("准确率:",right/oridata.shape[0]*100,"%")

