import torch
import torch.nn as nn
from ebrec.models.newsrec_pytorch.news_encoder import NewsEncoder
from ebrec.models.newsrec_pytorch.user_encoder import UserEncoder

class NRMSModel(nn.Module):
    def __init__(self, hparams_nrms, word2vec_embedding, seed):
        super(NRMSModel, self).__init__()
        self.news_encoder = NewsEncoder(hparams_nrms, word2vec_embedding, seed)
        self.candidate_encoder = NewsEncoder(hparams_nrms, word2vec_embedding, seed)
        self.user_encoder = UserEncoder(self.news_encoder, hparams_nrms, seed)
        # print("News encoder: ", self.news_encoder)
        # print("User encoder: ", self.user_encoder)
        self.hparams_nrms = hparams_nrms
    
    def forward(self, his_input_title, pred_input_title):
        batch_size, npratio, title_size = pred_input_title.shape
        
        user_present = self.user_encoder(his_input_title)
        user_present_unsqueezed = user_present.unsqueeze(1)
        pred_input_title_flat = pred_input_title.view(-1, title_size)
        news_present = self.candidate_encoder(pred_input_title_flat)
        news_present = news_present.view(batch_size, npratio, -1)
        # print("User present shape: ", user_present_unsqueezed.shape)
        # print("News present shape: ", news_present.shape)
        
        # apply inner product between user_present and news_present
        preds = torch.matmul(user_present_unsqueezed, news_present.transpose(1, 2)).squeeze(1)
        # print("Preds after fot shape: ", preds.shape)
        # apply softmax to get probability
        preds = torch.softmax(preds, dim=1)
        # print("Preds shape: ", preds.shape)

        return preds