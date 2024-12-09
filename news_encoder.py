import torch
import torch.nn as nn
from layers import AttLayer2, SelfAttention
import torch.nn.init as init

class NewsEncoder(nn.Module):
    def __init__(self, hparams_nrms, word2vec_embedding, seed):
        super(NewsEncoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word2vec_embedding, freeze=True)
        self.dropout = nn.Dropout(hparams_nrms.dropout)
        self.multihead_attention = nn.MultiheadAttention(hparams_nrms.embedded_dimension, hparams_nrms.head_num)
        self.attention = AttLayer2(hparams_nrms.attention_hidden_dim, seed)
        self.initialize_weights(seed)

    def initialize_weights(self, seed):
        """Inicializa explícitamente los pesos de las capas."""
        torch.manual_seed(seed)  # Garantizar reproducibilidad
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)  # Xavier para Linear
                if module.bias is not None:
                    init.zeros_(module.bias)  # Bias en 0
            elif isinstance(module, nn.MultiheadAttention):
                # Inicializar pesos y biases del MultiheadAttention
                init.xavier_uniform_(module.in_proj_weight)
                init.xavier_uniform_(module.out_proj.weight)
                if module.in_proj_bias is not None:
                    init.zeros_(module.in_proj_bias)
            elif isinstance(module, nn.Embedding):
                # Embeddings: distribución uniforme
                init.uniform_(module.weight, -0.1, 0.1)
        
    def forward(self, embedded_sequences):
        # Input size is (batch_size, num of news, title_size, embedded_dimension)
        embedded_sequences_reshaped = embedded_sequences.view(-1, embedded_sequences.size(2), embedded_sequences.size(3))
        y,_ = self.multihead_attention(embedded_sequences_reshaped, embedded_sequences_reshaped, embedded_sequences_reshaped)
        y = y.view(embedded_sequences.size(0), embedded_sequences.size(1), embedded_sequences.size(2), -1)
        # Output size is (batch_size, num of news, title_size, embedded_dimension)
        
        y = self.dropout(y)
        
        # Reshape for attention layer
        y = y.view(-1, y.size(2), y.size(3))
        # Output size is (batch_size * num of news, title_size, embedded_dimension)
        
        # Input size is (batch_size * num of news, title_size, embedded_dimension)
        news_representation = self.attention(y)
        # Output size is (batch_size * num of news, embedded_dimension)
        
        del embedded_sequences
        
        return news_representation