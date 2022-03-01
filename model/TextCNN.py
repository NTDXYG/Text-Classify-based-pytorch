import torch

from Config import class_number

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
import DataSet
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        Vocab = len(DataSet.getTEXT().vocab)  ## 已知词的数量
        Dim = 256  ##每个词向量长度
        Cla = class_number  ##类别数
        Ci = 1  ##输入的channel数
        Knum = 256  ## 每种卷积核的数量
        Ks = [2,3,4]  ## 卷积核list，形如[2,3,4]

        self.embed = nn.Embedding(Vocab, Dim)  ## 词向量，这里直接随机
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Knum, (K, Dim)) for K in Ks])  ## 卷积层
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(Ks) * Knum, Cla)  ##全连接层

    def forward(self, x):
        # [batch len, text size]
        x = self.embed(x)
        # [batch len, text size, emb dim]
        x = x.unsqueeze(1)
        #  [batch len, Ci, text size, emb dim]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # len(Ks)*[batch size, Knum, text len]
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]
        # len(Ks)*[batch size, Knum]
        x = torch.cat(x, 1)
        # [batch size, Knum*len(Ks)]
        x = self.dropout(x)
        # [batch size, Knum*len(Ks)]
        logit = self.fc(x)
        # [batch size, Cla]
        return logit