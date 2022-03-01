import torch

from Config import class_number

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
import DataSet
import torch.nn as nn
import torch.nn.functional as F

class TextRNN_Attention(nn.Module):
    def __init__(self):
        super(TextRNN_Attention, self).__init__()
        Vocab = len(DataSet.getTEXT().vocab)  ## 已知词的数量
        Dim = 256  ##每个词向量长度
        dropout = 0.2
        hidden_size = 256 #隐藏层数量
        num_classes = class_number ##类别数
        num_layers = 2 ##双层LSTM

        self.embedding = nn.Embedding(Vocab, Dim)  ## 词向量，这里直接随机
        self.lstm = nn.LSTM(Dim, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # [batch len, text size]
        x = self.embedding(x)
        # [batch size, text size, embedding]
        output, (hidden, cell) = self.lstm(x)
        # output = [batch size, text size, num_directions * hidden_size]
        M = self.tanh1(output)
        # [batch size, text size, num_directions * hidden_size]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        # [batch size, text size, 1]
        out = output * alpha
        # [batch size, text size, num_directions * hidden_size]
        out = torch.sum(out, 1)
        # [batch size, num_directions * hidden_size]
        out = F.relu(out)
        # [batch size, num_directions * hidden_size]
        out = self.fc(out)
        # [batch size, num_classes]
        return out