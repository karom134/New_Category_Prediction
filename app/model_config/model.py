import torch.nn as nn
import torch


class Category_classifier(nn.Module):
  def __init__(self,vocab_size,embedding_dim,hidden_dim):
    super(Category_classifier,self).__init__()
    
    self.embeddding=nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
    self.dropout=nn.Dropout(0.3)
    self.lstm=nn.LSTM(input_size=embedding_dim,hidden_size=hidden_dim,batch_first=True)
    self.linear=nn.Linear(hidden_dim,11)
    self.sigmoid=nn.Sigmoid()

  def forward(self,x):
    x=self.embeddding(x)
    x=self.dropout(x)
    lstm_out,(ht,ct)=self.lstm(x)
    x=self.linear(ht[-1])
    return self.sigmoid(x)