import torch
import torch.nn as nn
from news_encoder import NewsEncoder
from user_encoder import UserEncoder

class NRMSModel(nn.Module):
    def __init__(self, hparams_nrms, word2vec_embedding, seed):
        super(NRMSModel, self).__init__()
        self.candidate_encoder = NewsEncoder(hparams_nrms, word2vec_embedding, seed)
        self.user_encoder = UserEncoder(self.candidate_encoder, hparams_nrms, seed)
        self.hparams_nrms = hparams_nrms
        
    
    def forward(self, his_input_title, pred_input_title):
        # Get sizes
        batch_size, npratio, title_size = pred_input_title.shape
        pred_input_title_flat = pred_input_title.view(-1, title_size)
        
        # Input size is (batch_size, history_size, title_size)
        user_present = self.user_encoder(his_input_title)
        # Output size is (batch_size, hidden_dim)
        
        # Reshape for titleencoder: treat each news title independently
        user_present_unsqueezed = user_present.unsqueeze(1)
        
        # Input size is (batch_size * npratio, title_size)
        candidate_news_encoded = self.candidate_encoder(pred_input_title_flat).view(batch_size, npratio, -1)
        # Output size is (batch_size, npratio, hidden_dim)
        
        # apply inner product between user_present and news_present
        # Input size is (batch_size, 1, hidden_dim) and (batch_size, npratio, hidden_dim)
        preds = torch.matmul(user_present_unsqueezed, candidate_news_encoded.transpose(1, 2)).squeeze(1)
        # Output size is (batch_size, npratio)
        
        # apply softmax to get probability
        # Input size is (batch_size, npratio)
        preds = torch.softmax(preds, dim=1) if npratio > 1 else torch.sigmoid(preds)
        
        # Output is the probability of each news title

        return preds