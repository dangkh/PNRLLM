import copy

import torch
import torch.nn as nn
import numpy as np
from ..base.layers import *
from pathlib import Path

import copy

import torch
import torch.nn as nn
import numpy as np


class NewsEncoder(nn.Module):
    def __init__(self, embedding_matrix, num_category, num_subcategory, num_words_title):
        super(NewsEncoder, self).__init__()
        self.embedding_matrix = embedding_matrix
        self.drop_rate = 0.2
        self.num_words_title = num_words_title
        self.use_category = True
        self.use_subcategory = True
        category_emb_dim = 100
        news_dim = 400
        news_query_vector_dim = 200
        word_embedding_dim = 300
        self.category_emb = nn.Embedding(num_category + 1, category_emb_dim, padding_idx=0)
        self.category_dense = nn.Linear(category_emb_dim, news_dim)
        self.subcategory_emb = nn.Embedding(num_subcategory + 1, category_emb_dim, padding_idx=0)
        self.subcategory_dense = nn.Linear(category_emb_dim, news_dim)
        self.final_attn = AttentionPooling(news_dim, news_query_vector_dim)
        self.cnn = nn.Conv1d(
            in_channels=word_embedding_dim,
            out_channels=news_dim,
            kernel_size=3,
            padding=1
        )
        self.attn = AttentionPooling(news_dim, news_query_vector_dim)
        self.lnorm = nn.LayerNorm(news_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, mask=None):
        '''
            x: batch_size, word_num
            mask: batch_size, word_num
        '''
        title = torch.narrow(x, -1, 0, self.num_words_title).long()
        word_vecs = F.dropout(self.embedding_matrix(title),
                              p=self.drop_rate,
                              training=self.training)
        context_word_vecs = self.cnn(word_vecs.transpose(1, 2)).transpose(1, 2)
        # context_word_vecs = self.cnn(word_vecs.transpose(1, 2)).transpose(1, 2)
        # stop
        title_vecs = self.lnorm(self.attn(context_word_vecs, mask))
        # title_vecs = self.attn(context_word_vecs, mask)
        all_vecs = [title_vecs]

        start = self.num_words_title
        if self.use_category:
            category = torch.narrow(x, -1, start, 1).squeeze(dim=-1).long()
            category_vecs = self.category_dense(self.category_emb(category))
            all_vecs.append(category_vecs)
            start += 1
        if self.use_subcategory:
            subcategory = torch.narrow(x, -1, start, 1).squeeze(dim=-1).long()
            subcategory_vecs = self.subcategory_dense(self.subcategory_emb(subcategory))
            all_vecs.append(subcategory_vecs)

        if len(all_vecs) == 1:
            news_vecs = all_vecs[0]
        else:
            all_vecs = torch.stack(all_vecs, dim=1)
            
            news_vecs = self.final_attn(all_vecs)
        return self.act(news_vecs)
        # return news_vecs

class NewsEncoder_PNR(nn.Module):
    def __init__(self, embedding_matrix, num_category, num_subcategory, num_words_title):
        super(NewsEncoder_PNR, self).__init__()
        self.embedding_matrix = embedding_matrix
        self.drop_rate = 0.2
        self.num_words_title = num_words_title
        self.use_category = True
        self.use_subcategory = True
        category_emb_dim = 100
        news_dim = 400
        news_query_vector_dim = 200
        word_embedding_dim = 300
        self.category_emb = nn.Embedding(num_category + 1, category_emb_dim, padding_idx=0)
        self.category_dense = nn.Linear(category_emb_dim, news_dim)
        self.subcategory_emb = nn.Embedding(num_subcategory + 1, category_emb_dim, padding_idx=0)
        self.subcategory_dense = nn.Linear(category_emb_dim, news_dim)
        self.final_attn = AttentionPooling(news_dim, news_query_vector_dim)
        self.cnn = nn.Conv1d(
            in_channels=word_embedding_dim,
            out_channels=news_dim,
            kernel_size=3,
            padding=1
        )
        self.attn = AttentionPooling(news_dim, news_query_vector_dim)
        self.cln = nn.Linear(300,400)
        self.lnorm = nn.LayerNorm(news_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, mask=None):
        '''
            x: batch_size, word_num
            mask: batch_size, word_num
        '''
        title = torch.narrow(x, -1, 0, self.num_words_title).long()
        word_vecs = F.dropout(self.embedding_matrix(title),
                              p=self.drop_rate,
                              training=self.training)
        context_word_vecs = self.cnn(word_vecs.transpose(1, 2)).transpose(1, 2)
        # context_word_vecs = self.cnn(word_vecs.transpose(1, 2)).transpose(1, 2)
        # stop
        title_vecs = self.lnorm(self.attn(context_word_vecs, mask))
        all_vecs = [title_vecs]

        start = self.num_words_title
        if self.use_category:
            category = torch.narrow(x, -1, start, 1).squeeze(dim=-1).long()
            category_vecs = self.category_dense(self.category_emb(category))
            all_vecs.append(category_vecs)
            start += 1
        if self.use_subcategory:
            subcategory = torch.narrow(x, -1, start, 1).squeeze(dim=-1).long()
            subcategory_vecs = self.subcategory_dense(self.subcategory_emb(subcategory))
            all_vecs.append(subcategory_vecs)

        if len(all_vecs) == 1:
            news_vecs = all_vecs[0]
        else:
            all_vecs = torch.stack(all_vecs, dim=1)
            
            news_vecs = self.final_attn(all_vecs)
        return self.act(news_vecs)

class NewsEncoder_NRMS(nn.Module):
    def __init__(self, args, embedding_matrix):
        super(NewsEncoder_NRMS, self).__init__()
        self.embedding_matrix = embedding_matrix
        self.drop_rate = 0.2
        self.news_dim = 400
        self.num_attention_heads = 4
        self.word_embedding_dim = 300
        self.news_query_vector_dim = 200
        self.dim_per_head = self.news_dim // self.num_attention_heads
        assert self.news_dim == self.num_attention_heads * self.dim_per_head
        self.multi_head_self_attn = MultiHeadSelfAttention(
            self.word_embedding_dim,
            self.num_attention_heads,
            self.dim_per_head,
            self.dim_per_head
        )
        self.attn = AttentionPooling(self.news_dim, self.news_query_vector_dim)

    def forward(self, x, mask=None):
        '''
            x: batch_size, word_num
            mask: batch_size, word_num
        '''
        word_vecs = F.dropout(self.embedding_matrix(x.long()),
                              p=self.drop_rate,
                              training=self.training)
        multihead_text_vecs = self.multi_head_self_attn(word_vecs, word_vecs, word_vecs, mask)
        multihead_text_vecs = F.dropout(multihead_text_vecs,
                                        p=self.drop_rate,
                                        training=self.training)
        news_vec = self.attn(multihead_text_vecs, mask)
        return news_vec