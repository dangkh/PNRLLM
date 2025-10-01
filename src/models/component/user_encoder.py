import torch
import torch.nn as nn
import numpy as np
from ..base.layers import *

class UserEncoder(nn.Module):
    def __init__(self, cfg):
        super(UserEncoder, self).__init__()
        news_dim = 400
        user_query_vector_dim = 200
        self.user_log_length = cfg.his_size
        self.user_log_mask = False
        self.attn = AttentionPooling(news_dim, user_query_vector_dim)
        self.pad_doc = nn.Parameter(torch.empty(1, news_dim).uniform_(-1, 1)).type(torch.FloatTensor)

    def forward(self, news_vecs, log_mask=None):
        '''
            news_vecs: batch_size, history_num, news_dim
            log_mask: batch_size, history_num
        '''
        bz = news_vecs.shape[0]
        if self.user_log_mask:
            user_vec = self.attn(news_vecs, log_mask)
        else:
            padding_doc = self.pad_doc.unsqueeze(dim=0).expand(bz, self.user_log_length, -1)
            news_vecs = news_vecs * log_mask.unsqueeze(dim=-1) + padding_doc * (1 - log_mask.unsqueeze(dim=-1))
            user_vec = self.attn(news_vecs)
        return user_vec

class UserEncoder_NRMS(nn.Module):
    def __init__(self, args):
        super(UserEncoder_NRMS, self).__init__()
        self.args = args
        self.news_dim = 400
        self.user_query_vector_dim = 200
        self.user_log_length = 50
        self.user_log_mask = False
        self.num_attention_heads = 4
        self.dim_per_head = self.news_dim // self.num_attention_heads
        assert self.news_dim == self.num_attention_heads * self.dim_per_head
        self.multi_head_self_attn = MultiHeadSelfAttention(self.news_dim, self.num_attention_heads, self.dim_per_head, self.dim_per_head)
        self.attn = AttentionPooling(self.news_dim, self.user_query_vector_dim)
        self.pad_doc = nn.Parameter(torch.empty(1, self.news_dim).uniform_(-1, 1)).type(torch.FloatTensor)

    def forward(self, news_vecs, log_mask=None):
        '''
            news_vecs: batch_size, history_num, news_dim
            log_mask: batch_size, history_num
        '''
        bz = news_vecs.shape[0]
        if self.user_log_mask:
            news_vecs = self.multi_head_self_attn(news_vecs, news_vecs, news_vecs, log_mask)
            user_vec = self.attn(news_vecs, log_mask)
        else:
            padding_doc = self.pad_doc.unsqueeze(dim=0).expand(bz, self.user_log_length, -1)
            news_vecs = news_vecs * log_mask.unsqueeze(dim=-1) + padding_doc * (1 - log_mask.unsqueeze(dim=-1))
            news_vecs = self.multi_head_self_attn(news_vecs, news_vecs, news_vecs)
            user_vec = self.attn(news_vecs)
        return user_vec
