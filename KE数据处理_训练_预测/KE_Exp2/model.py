import torch
import torch.nn as nn




# 定义全连接神经网络(FCNN)模型
class FCNN(nn.Module):
    def __init__(self,fetlen,lablen,cuda=True):
        super(FCNN, self).__init__()
        self.fetlen = fetlen
        self.lablen = lablen
        self.cuda = cuda
        self.fc1 = nn.Linear(self.fetlen, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        self.fc5 = nn.Linear(in_features=64, out_features=32)
        self.fc6 = nn.Linear(in_features=32,out_features=self.lablen)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x
