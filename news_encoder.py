import torch
import torch.nn as nn
from ebrec.models.newsrec_pytorch.layers import AttLayer2, SelfAttention

class NewsEncoder(nn.Module):
    def __init__(self, hparams_nrms, word2vec_embedding, seed):
        super(NewsEncoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word2vec_embedding, freeze=False)
        self.dropout = nn.Dropout(hparams_nrms.dropout)
        # self.self_attention = SelfAttention(hparams_nrms.head_num, hparams_nrms.head_dim, seed)
        self.multihead_attention = nn.MultiheadAttention(hparams_nrms.embedded_dimension, hparams_nrms.head_num)
        self.attention = AttLayer2(hparams_nrms.attention_hidden_dim, seed)
        
    def forward(self, sequences_input_title):
        embedded_sequences = self.embedding(sequences_input_title)
        # print("embedded sequence size: ", embedded_sequences.shape)
        y = self.dropout(embedded_sequences)
        # print(y.shape)
        y = self.multihead_attention(y, y, y)
        # print("News encoder multihead attention output: ", y[0].shape)
        # print(y.shape)
        y = self.dropout(y[0])
        # print(y.shape)
        pred_title = self.attention(y)
        # print("pred title after attention in news encoder: ", pred_title.shape)
        return pred_title