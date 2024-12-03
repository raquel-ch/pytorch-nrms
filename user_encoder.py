import torch
import torch.nn as nn
from ebrec.models.newsrec_pytorch.layers import AttLayer2, SelfAttention
from ebrec.models.newsrec_pytorch.model_config import hparams_nrms

class UserEncoder(nn.Module):
    def __init__(self, news_encoder, hparams_nrms, seed):
        super(UserEncoder, self).__init__()
        self.news_encoder = news_encoder
        self.multihead_attention = nn.MultiheadAttention(hparams_nrms.embedded_dimension, hparams_nrms.head_num)
        self.additive_attention = AttLayer2(hparams_nrms.attention_hidden_dim, seed)
        self.dropout = nn.Dropout(hparams_nrms.dropout)
        
    def forward(self, his_input_title):
        batch_size, history_size, title_size = his_input_title.shape
        
        # Reshape for titleencoder: treat each news title independently
        his_input_title_flat = his_input_title.view(-1, title_size)  # Shape: (batch_size * history_size, title_size)
        click_title_presents = self.news_encoder(his_input_title_flat)  # Shape: (batch_size * history_size, hidden_dim)
        
        # Reshape back to include history_size
        click_title_presents = click_title_presents.view(batch_size, history_size, -1)  # Shape: (batch_size, history_size, hidden_dim)
        # print("Click title presents: ", click_title_presents.shape)
        # Self-attention over the historical clicked news representations
        y = self.multihead_attention(click_title_presents, click_title_presents, click_title_presents)  # Shape: (batch_size, history_size, hidden_dim)
        # print("Multihead attention output from user encoder: ", y[0].shape)
        # Dropout
        y = self.dropout(y[0])
        
        # Attention layer for user representation
        user_present = self.additive_attention(y) 
        # print("User encoder output after additive: ", user_present.shape)
        
        return user_present
        