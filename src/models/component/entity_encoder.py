import torch
import torch.nn as nn
from ..base.layers import *

class EntityEncoder(nn.Module):
    def __init__(self, cfg):
        super(EntityEncoder, self).__init__()

        self.entity_dim = cfg.entity_emb_dim
        self.news_dim = 400

        self.mhat = MultiHeadSelfAttention(self.entity_dim, cfg.head_num , cfg.head_dim , cfg.head_dim)
        self.mhat_dim = cfg.head_num * cfg.head_dim
        self.lnorm = nn.LayerNorm(self.mhat_dim)
        self.drop = nn.Dropout(p=cfg.dropout_probability)

        self.att = AttentionPooling(self.mhat_dim, cfg.attention_hidden_dim)
        self.ln = nn.Linear(self.mhat_dim, self.news_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, entity_input, entity_mask=None):

        batch_size, num_news, num_entity, _ = entity_input.shape
        entity_input = entity_input.view(-1, num_entity, self.entity_dim)
        entity_input = self.drop(entity_input)
        entity_input = self.mhat(entity_input, entity_input, entity_input)
        entity_input = self.lnorm(entity_input)

        entity_input = self.drop(entity_input)
        entity_input = self.att(entity_input)
        entity_input = self.lnorm(entity_input)
        entity_input = self.ln(entity_input)
        entity_input = self.act(entity_input)
        
        return  entity_input.view(batch_size, num_news, -1)        