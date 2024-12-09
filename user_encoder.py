import torch
import torch.nn as nn
from layers import AttLayer2, SelfAttention
from model_config import hparams_nrms
import torch.nn.init as init

class UserEncoder(nn.Module):
    def __init__(self, hparams_nrms, seed):
        super(UserEncoder, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(hparams_nrms.embedded_dimension, hparams_nrms.head_num)
        self.additive_attention = AttLayer2(hparams_nrms.attention_hidden_dim, seed)
        self.dropout = nn.Dropout(hparams_nrms.dropout)
        self.initialize_weights(seed)
        self.batch_norm_attention = nn.BatchNorm1d(hparams_nrms.embedded_dimension)
        
    def initialize_weights(self, seed):
        """Inicializa explícitamente los pesos de las capas."""
        torch.manual_seed(seed)  # Garantizar reproducibilidad
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)  # Xavier para Linear
                if module.bias is not None:
                    init.zeros_(module.bias)  # Sesgo a 0
            elif isinstance(module, nn.MultiheadAttention):
                # Inicializar pesos y biases del MultiheadAttention
                init.xavier_uniform_(module.in_proj_weight)
                init.xavier_uniform_(module.out_proj.weight)
                if module.in_proj_bias is not None:
                    init.zeros_(module.in_proj_bias)
            elif isinstance(module, AttLayer2):
                # Las inicializaciones ya están en el constructor de AttLayer2
                pass
        
    def forward(self, encoded_news):
        all_news, embedding_size = encoded_news.shape
        num_of_users = all_news // hparams_nrms.history_size
        
        # Reshape from (all news, embedding size) to (batch_size, num of news, embedding size)
        click_title_presents = encoded_news.view(num_of_users, hparams_nrms.history_size, embedding_size)
        # Output size is (batch_size, num of news, embedding size)

        # Input size is (batch_size, num of news, embedding size)
        y, _ = self.multihead_attention(click_title_presents, click_title_presents, click_title_presents)  # Shape: (batch_size, history_size, hidden_dim)
        # Output size is (batch_size, num of news, embedding size)
        
        y = self.dropout(y)
        
        # Attention layer for user representation
        # Input size is (batch_size, num of news, embedding size)
        user_present = self.additive_attention(y) 
        # Output size is (batch_size, embedding size)
        
        
        return user_present
        