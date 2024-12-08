import torch
import torch.nn as nn
from layers import AttLayer2, SelfAttention
from model_config import hparams_nrms
import torch.nn.init as init

class UserEncoder(nn.Module):
    def __init__(self, news_encoder, hparams_nrms, seed):
        super(UserEncoder, self).__init__()
        self.news_encoder = news_encoder
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
        
    def forward(self, his_input_title):
        batch_size, history_size, title_size = his_input_title.shape
        # Reshape for titleencoder: treat each news title independently
        his_input_title_flat = his_input_title.view(-1, title_size)  # Shape: (batch_size * history_size, title_size)
        # Input size is (batch_size * history_size, title_size)
        click_title_presents = self.news_encoder(his_input_title_flat)  # Shape: (batch_size * history_size, hidden_dim)
        # Output size is (batch_size * history_size, hidden_dim)

        # Reshape back to include history_size
        click_title_presents = click_title_presents.view(batch_size, history_size, -1)  # Shape: (batch_size, history_size, hidden_dim)
        # Self-attention over the historical clicked news representations
        
        # Input size is (batch_size, history_size, hidden_dim)
        y, _ = self.multihead_attention(click_title_presents, click_title_presents, click_title_presents)  # Shape: (batch_size, history_size, hidden_dim)
        # Output size is (batch_size, history_size, hidden_dim)
        
        y = self.dropout(y)
        
        # y = self.batch_norm_attention(y[0])
        y = self.batch_norm_attention(y.permute(0, 2, 1)).permute(0,2,1)

        # Attention layer for user representation
        # Input size is (batch_size, history_size, hidden_dim)
        user_present = self.additive_attention(y) 
        # Output size is (batch_size, hidden_dim)
        
        return user_present
        