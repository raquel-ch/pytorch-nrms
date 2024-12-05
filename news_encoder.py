import torch
import torch.nn as nn
from layers import AttLayer2, SelfAttention
import torch.nn.init as init

class NewsEncoder(nn.Module):
    def __init__(self, hparams_nrms, word2vec_embedding, seed):
        super(NewsEncoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word2vec_embedding, freeze=False)
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
        
    def forward(self, sequences_input_title):
        # Input size is (batch_size, title_size)
        embedded_sequences = self.embedding(sequences_input_title).detach()
        # Output size is (batch_size, title_size, embedded_dimension)
        
        y = self.dropout(embedded_sequences)
        del embedded_sequences
        
        # Input size is (title_size, batch_size, embedded_dimension)
        y = self.multihead_attention(y, y, y)
        # Output size is (title_size, batch_size, embedded_dimension)
        y = self.dropout(y[0])
        
        # Input size is (batch_size, title_size, embedded_dimension)
        pred_title = self.attention(y)
        # Output size is (batch_size, embedded_dimension)
        
        return pred_title