import torch
import torch.nn as nn
from torch.nn import ReLU

from .base.layers import *
from .component.user_encoder import *
from .component.entity_encoder import EntityEncoder
from .component.news_encoder import *
import torch.nn.functional as F


class PNRLLM(torch.nn.Module):
    def __init__(self, embedding_matrix, entity_emb, num_category, num_subcategory, cfg):
        super(PNRLLM, self).__init__()
        self.news_dim = 400
        self.entity_dim = 100
        self.npratio = 4
        self.user_log_length = cfg.his_size
        self.cfg = cfg
        pretrained_word_embedding = torch.from_numpy(embedding_matrix).float()
        word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding,
                                                      freeze=False,
                                                      padding_idx=0)
        
        self.news_encoder = NewsEncoder_PNR( word_embedding, num_category, num_subcategory, cfg.title_size)
        self.user_encoder = UserEncoder(cfg)
        
        pretrain = torch.from_numpy(entity_emb).float()
        self.entity_embedding_layer = nn.Embedding.from_pretrained(pretrain, 
                                                                   freeze=False, 
                                                                   padding_idx=0)
        self.user_att = AttentionPooling(self.news_dim, cfg.attention_hidden_dim)
        self.candi_att = AttentionPooling(self.news_dim, cfg.attention_hidden_dim)
        self.entity_encoder = EntityEncoder(cfg)


        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, args):
        '''
            history: batch_size, history_length, num_word_title
            history_mask: batch_size, history_length
            candidate: batch_size, 1+K, num_word_title
            label: batch_size, 1+K
        '''
        history, history_mask, candidate, label = args
        
        e_his = history[:,:,-5:]
        history = history[:,:,:-5]
        e_candi = candidate[:,:,-5:]
        candidate = candidate[:,:,:-5]
        
        num_words = history.shape[-1]
        
        
        e_his = self.entity_embedding_layer(e_his)
        e_candi = self.entity_embedding_layer(e_candi)
        e_his = self.entity_encoder(e_his, None)
        e_candi = self.entity_encoder(e_candi, None)

        candidate = candidate.reshape(-1, num_words)
        candidate = self.news_encoder(candidate).reshape(-1, 1 + self.npratio, self.news_dim)
        history = history.reshape(-1, num_words)
        user_vec = self.news_encoder(history).reshape(-1, self.user_log_length, self.news_dim)
        
        candidate = self.candi_att(torch.stack([candidate, e_candi], dim=2).view(-1, 2, self.news_dim))
        candidate = candidate.view(-1, self.npratio+1, self.news_dim)
        user_vec = self.user_att(torch.stack([user_vec, e_his], dim=2).view(-1, 2, self.news_dim))
        user_vec = user_vec.view(-1, self.user_log_length, self.news_dim)
            
        user_vec = self.user_encoder(user_vec, history_mask)

        score = torch.bmm(candidate, user_vec.unsqueeze(dim=-1)).squeeze(dim=-1)
        loss = self.loss_fn(score, label)
  
        return loss, score


class NAML(torch.nn.Module):
    def __init__(self, embedding_matrix, entity_emb, num_category, num_subcategory, cfg):
        super(NAML, self).__init__()
        self.news_dim = 400
        self.entity_dim = 100
        self.npratio = 4
        self.user_log_length = cfg.his_size
        self.cfg = cfg
        pretrained_word_embedding = torch.from_numpy(embedding_matrix).float()
        word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding,
                                                      freeze=False,
                                                      padding_idx=0)
        
        self.news_encoder = NewsEncoder( word_embedding, num_category, num_subcategory, cfg.title_size)
        self.user_encoder = UserEncoder(cfg)
        
        if cfg.use_entity:
            pretrain = torch.from_numpy(entity_emb).float()
            self.entity_embedding_layer = nn.Embedding.from_pretrained(pretrain, 
                                                                       freeze=False, 
                                                                       padding_idx=0)
            self.user_att = AttentionPooling(self.news_dim, cfg.attention_hidden_dim)
            self.candi_att = AttentionPooling(self.news_dim, cfg.attention_hidden_dim)
            self.entity_encoder = EntityEncoder(cfg)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, args):
        '''
            history: batch_size, history_length, num_word_title
            history_mask: batch_size, history_length
            candidate: batch_size, 1+K, num_word_title
            label: batch_size, 1+K
        '''
        
        history, history_mask, candidate, label = args
        
        num_words = history.shape[-1]
        

        candidate = candidate.reshape(-1, num_words)
        candidate = self.news_encoder(candidate).reshape(-1, 1 + self.npratio, self.news_dim)
        history = history.reshape(-1, num_words)
        user_vec = self.news_encoder(history).reshape(-1, self.user_log_length, self.news_dim)
        
        user_vec = self.user_encoder(user_vec, history_mask)

        score = torch.bmm(candidate, user_vec.unsqueeze(dim=-1)).squeeze(dim=-1)
        loss = self.loss_fn(score, label)
  
        return loss, score        


class NRMS(torch.nn.Module):
    def __init__(self, embedding_matrix, entity_emb, num_category, num_subcategory, cfg):
        super(NRMS, self).__init__()
        self.args = cfg
        self.news_dim = 400
        self.npratio = 4
        self.user_log_length = cfg.his_size
        pretrained_word_embedding = torch.from_numpy(embedding_matrix).float()
        word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding,
                                                      freeze=False,
                                                      padding_idx=0)

        self.news_encoder = NewsEncoder_NRMS(cfg, word_embedding)
        self.user_encoder = UserEncoder_NRMS(cfg)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, args):
        '''
            history: batch_size, history_length, num_word_title
            history_mask: batch_size, history_length
            candidate: batch_size, 1+K, num_word_title
            label: batch_size, 1+K
        '''

        _, history, history_mask, candidate, label = args
        num_words = history.shape[-1]
        candidate_news = candidate.reshape(-1, num_words)
        candidate_news_vecs = self.news_encoder(candidate_news).reshape(-1, 1 + self.npratio, self.news_dim)

        history_news = history.reshape(-1, num_words)
        history_news_vecs = self.news_encoder(history_news).reshape(-1, self.user_log_length, self.news_dim)

        user_vec = self.user_encoder(history_news_vecs, history_mask)
        score = torch.bmm(candidate_news_vecs, user_vec.unsqueeze(dim=-1)).squeeze(dim=-1)
        loss = self.loss_fn(score, label)
        return loss, score


class UserEncoder_LSTUR(torch.nn.Module):
    def __init__(self, config):
        super(UserEncoder_LSTUR, self).__init__()
        self.num_filters = 300

        assert int(self.num_filters * 1.5) == self.num_filters * 1.5
        self.gru = nn.GRU(
            self.num_filters * 3,
            self.num_filters * 3)

    def forward(self, user, clicked_news_length, clicked_news_vector):
        """
        Args:
            user:
                ini: batch_size, num_filters * 3
                con: batch_size, num_filters * 1.5
            clicked_news_length: batch_size,
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters * 3
        Returns:
            (shape) batch_size, num_filters * 3
        """
        clicked_news_length[clicked_news_length == 0] = 1
        # 1, batch_size, num_filters * 3
        packed_clicked_news_vector = pack_padded_sequence(
                clicked_news_vector,
                clicked_news_length,
                batch_first=True,
                enforce_sorted=False)
        _, last_hidden = self.gru(packed_clicked_news_vector,
                                  user.unsqueeze(dim=0))
        return last_hidden.squeeze(dim=0)

class LSTUR(torch.nn.Module):
    """
    LSTUR network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self, embedding_matrix, entity_emb, num_category, num_subcategory, cfg):
        """
        # ini
        user embedding: num_filters * 3
        news encoder: num_filters * 3
        GRU:
        input: num_filters * 3
        hidden: num_filters * 3

        # con
        user embedding: num_filter * 1.5
        news encoder: num_filters * 3
        GRU:
        input: num_fitlers * 3
        hidden: num_filter * 1.5
        embedding_matrix, entity_emb, num_category, num_subcategory, cfg
        """
        super(LSTUR, self).__init__()
        self.config = cfg
        self.num_users =  1+50000
        self.num_filters = 300
        self.news_dim = 400
        self.npratio = 4
        self.user_log_length = 50
        pretrained_word_embedding = torch.from_numpy(embedding_matrix).float()
        word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding,
                                                      freeze=False,
                                                      padding_idx=0)

        self.news_encoder = NewsEncoder_NRMS(cfg, word_embedding)

        self.user_encoder = UserEncoder_LSTUR(cfg)
        self.user_embedding = nn.Embedding(
            self.num_users,
            self.news_dim,
            padding_idx=0)

        self.gru = nn.GRU(400, 400, batch_first=True)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, args):
        """
        Args:
            user: batch_size,
            clicked_news_length: batch_size,
            candidate_news:
                [
                    {
                        "category": batch_size,
                        "subcategory": batch_size,
                        "title": batch_size * num_words_title
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "category": batch_size,
                        "subcategory": batch_size,
                        "title": batch_size * num_words_title
                    } * num_clicked_news_a_user
                ]
                user, clicked_news_length, candidate_news, clicked_news
        Returns:
            click_probability: batch_size
        """
        # batch_size, 1 + K, num_filters * 3

        _, meta, history, history_mask, candidate, label = args
        num_words = history.shape[-1]
        candidate_news = candidate.reshape(-1, num_words)
        candidate_news_vecs = self.news_encoder(candidate_news).reshape(-1, 1 + self.npratio, self.news_dim)

        history_news = history.reshape(-1, num_words)
        history_news_vecs = self.news_encoder(history_news).reshape(-1, self.user_log_length, self.news_dim)

        long_term_user_repr = self.user_embedding(meta[:,0])
        _, short_term_user_repr = self.gru(history_news_vecs)
        short_term_user_repr = short_term_user_repr.squeeze(0)

        user_vec = F.normalize(long_term_user_repr + short_term_user_repr, p=2, dim=1)

        # user_vec = self.user_encoder(history_news_vecs, history_mask)
        score = torch.bmm(candidate_news_vecs, user_vec.unsqueeze(dim=-1)).squeeze(dim=-1)
        loss = self.loss_fn(score, label)

        return loss, score
    