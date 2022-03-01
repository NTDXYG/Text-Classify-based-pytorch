import torch
import torch.nn as nn
import torch.nn.functional as F
import DataSet
from Config import class_number


class TextRCNN(nn.Module):
    def __init__(self,
                 vocab_size = len(DataSet.getTEXT().vocab),  # 词典的大小(总共有多少个词语/字)
                 n_class = class_number,  # 分类的类型
                 embed_dim=256,  # embedding的维度
                 rnn_hidden=256,
                 dropout=0.2
                 ):
        super(TextRCNN, self).__init__()
        self.rnn_hidden = rnn_hidden
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )
        self.maxpool = nn.MaxPool1d(256)
        self.lstm = nn.LSTM(embed_dim, rnn_hidden, 2,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(in_features=embed_dim+2*rnn_hidden,
                            out_features=n_class)

    def forward(self, x):
        #[batch, text_len]
        x = self.embedding(x)
        # [batch, text_len, embed_dim]
        # output, h_n = self.gru(x)
        output, _ = self.lstm(x)
        x = torch.cat([x, output], dim=2)
        # [batch, text_len, 2*rnn_hidden+embed_dim]
        x = F.relu(x)
        x = x.permute(0, 2, 1)
        x = self.maxpool(x).squeeze()
        # x = F.max_pool2d(x, (x.shape[1], 1))
        # [batch, 1, 2*rnn_hidden+embed_dim]
        # x = x.reshape(-1,2 * self.rnn_hidden+self.embed_dim)
        # [batch, 2*rnn_hidden+embed_dim]
        x = self.fc(x)   # [batch, n_class]
        # x = torch.sigmoid(x)
        return x