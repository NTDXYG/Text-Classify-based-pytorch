import jieba
import torch
import DataSet
import re
from Config import fix_length, label_list

def x_tokenize(x):
    str = re.sub('[^\u4e00-\u9fa5]', "", x)
    return jieba.lcut(str)

def getModel(name):
    model = torch.load('done_model/'+name+'_model.pkl')
    return model

model = getModel('Transformer')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
sent1 = '风险评估：时间为2002。描述为Unreal IRCd 3.1.1版本cio_main.c中的 Cio_PrintF函数存在格式字符串漏洞。远程攻击者可以通过格式字符串说明符导致服务拒绝（崩溃）和可能执行任意代码。'

def predict_sentiment(model, sentence):
    model.eval()
    tokenized = x_tokenize(sentence)
    indexed = [DataSet.getTEXT().vocab.stoi[t] for t in tokenized]
    if(len(indexed)>fix_length):
        indexed = indexed[:fix_length]
    else:
        for i in range(fix_length-len(indexed)):
            indexed.append(DataSet.getTEXT().vocab.stoi['<pad>'])
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    tensor = torch.t(tensor)
    prediction = torch.argmax(model(tensor), dim=1)
    return prediction.item()

print(label_list[predict_sentiment(model, sent1)])