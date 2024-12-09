import torch
import torch.nn as nn
from news_encoder import NewsEncoder
from user_encoder import UserEncoder

class NRMSModel(nn.Module):
    def __init__(self, hparams_nrms, word2vec_embedding, seed):
        super(NRMSModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word2vec_embedding, freeze=True)
        self.encode_news = NewsEncoder(hparams_nrms, word2vec_embedding, seed)
        self.user_encoder = UserEncoder(hparams_nrms, seed)
        self.hparams_nrms = hparams_nrms
        
    
    def forward(self, his_input_title, pred_input_title):
        # Get sizes
        batch_size, npratio, title_size = pred_input_title.shape
        embedding_size = self.hparams_nrms.embedded_dimension
        
        browsed_news_embedding = self.embedding(his_input_title).detach()
        candidate_news_embedding = self.embedding(pred_input_title).detach()
        
        clicked_news_representation = self.encode_news(browsed_news_embedding)
        candidate_news_representation = self.encode_news(candidate_news_embedding).view(batch_size, npratio, embedding_size)
        
        # Input size is (batch_size * history_size, embedding size)
        user_present = self.user_encoder(clicked_news_representation)
        # Output size is (batch_size, embedding size)
        
        # Reshape for titleencoder: treat each news title independently
        user_present_unsqueezed = user_present.unsqueeze(1)
        
        del browsed_news_embedding, candidate_news_embedding
        
        # apply inner product between user_present and news_present
        # Input size is (batch_size, embedding size) and (batch_size * npratio, embedding size)
        preds = torch.matmul(user_present_unsqueezed, candidate_news_representation.transpose(1,2)).squeeze()
        
        # apply softmax to get probability
        # Input size is (batch_size, npratio)
        preds = torch.softmax(preds, dim=1) if npratio > 1 else torch.sigmoid(preds)
        # Output is the probability of each news title

        return preds