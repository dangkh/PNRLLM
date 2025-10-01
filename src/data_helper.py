import os
import random
import numpy as np
import torch
from typing import Callable
import logging
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score
import collections
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
    
import torch.nn.functional as F
from tqdm import tqdm
import random
import pickle
from collections import Counter
import json
import itertools
from pathlib import Path

def update_dict(target_dict, key, value=None):
    """
    Function for updating dict with key / key+value

    Args:
        target_dict(dict): target dict
        key(string): target key
        value(Any, optional): if None, equals len(dict+1)
    """
    if key not in target_dict:
        if value is None:
            target_dict[key] = len(target_dict) + 1
        else:
            target_dict[key] = value


def get_sample(all_elements, num_sample):
    if num_sample > len(all_elements):
        return random.sample(all_elements * (num_sample // len(all_elements) + 1), num_sample)
    else:
        return random.sample(all_elements, num_sample)

def prepare_distributed_data(cfg, mode = "train"):
    # check
    target_file = os.path.join(cfg.data_dir + f'_{mode}', f"behaviors_np{cfg.npratio}_0.tsv")
    if os.path.exists(target_file) and not cfg.reprocess:
        return 0
    print(f'Target_file is not exist. New behavior file in {target_file}')
    behaviors = []
    behavior_file_path = os.path.join(cfg.data_dir + f'_{mode}', 'behaviors.tsv')
    print(target_file, behavior_file_path)

    if mode == 'train':
        with open(behavior_file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                iid, uid, time, history, imp = line.strip().split('\t')
                
                # his = history.split(' ')
                # # print(his, len(his))
                # if his[0] == '':
                #     continue

                impressions = [x.split('-') for x in imp.split(' ')]
                pos, neg = [], []
                for news_ID, label in impressions:
                    if label == '0':
                        neg.append(news_ID)
                    elif label == '1':
                        pos.append(news_ID)
                if len(pos) == 0 or len(neg) == 0:
                    continue
                for pos_id in pos:
                    neg_candidate = get_sample(neg, cfg.npratio)
                    neg_str = ' '.join(neg_candidate)
                    new_line = '\t'.join([iid, uid, time, history, pos_id, neg_str]) + '\n'
                    behaviors.append(new_line)
        random.shuffle(behaviors)

        behaviors_per_file = [[] for _ in range(cfg.gpu_num)]
        for i, line in enumerate(behaviors):
            behaviors_per_file[i % cfg.gpu_num].append(line)

    elif mode in ['val', 'test']:
        behaviors_per_file = [[] for _ in range(cfg.gpu_num)]
        behavior_file_path = os.path.join(cfg.data_dir + f'_{mode}', 'behaviors.tsv')
        with open(behavior_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f)):
                behaviors_per_file[i % cfg.gpu_num].append(line)

    print(f'[{mode}]Writing files...')
    for i in range(cfg.gpu_num):
        processed_file_path = os.path.join(cfg.data_dir + f'_{mode}', f'behaviors_np{cfg.npratio}_{i}.tsv')
        with open(processed_file_path, 'w') as f:
            f.writelines(behaviors_per_file[i])

    return len(behaviors)

def read_parsed_news(cfg, news, news_dict,
                     category_dict=None, subcategory_dict=None, entity_dict=None,
                     word_dict=None):
    news_num = len(news) + 1
    news_category, news_subcategory, news_index = [np.zeros((news_num, 1), dtype='int32') for _ in range(3)]
    news_entity = np.zeros((news_num, cfg.entity_size), dtype='int32')

    news_title = np.zeros((news_num, cfg.title_size), dtype='int32')

    for _news_id in tqdm(news, total=len(news), desc="Processing parsed news"):
        _title, _category, _subcategory, _entity_ids, _news_index = news[_news_id]

        news_category[_news_index, 0] = category_dict[_category] if _category in category_dict else 0
        news_subcategory[_news_index, 0] = subcategory_dict[_subcategory] if _subcategory in subcategory_dict else 0
        news_index[_news_index, 0] = news_dict[_news_id]

        # entity
        entity_index = [entity_dict[entity_id] if entity_id in entity_dict else 0 for entity_id in _entity_ids]
        news_entity[_news_index, :min(cfg.entity_size, len(_entity_ids))] = entity_index[:cfg.entity_size]

        for _word_id in range(min(cfg.title_size, len(_title))):
            if _title[_word_id] in word_dict:
                news_title[_news_index, _word_id] = word_dict[_title[_word_id]]

    if cfg.backbones == "PNRLLM":
        return news_title, news_category, news_subcategory, news_entity
    if cfg.backbones == 'NRMS':
        return news_title
    return news_title, news_category, news_subcategory


def read_raw_news(cfg, file_path, mode='train'):

    data_dir = {"train": cfg.data_dir + '_train', "val": cfg.data_dir + '_val', "test": cfg.data_dir + '_test'}

    if mode in ['val', 'test']:
        news_dict = pickle.load(open(Path(data_dir["train"]) / "news_dict.bin", "rb"))
        entity_dict = pickle.load(open(Path(data_dir["train"]) / "entity_dict.bin", "rb"))
        news = pickle.load(open(Path(data_dir["train"]) / "nltk_news.bin", "rb"))
    else:
        news = {}
        news_dict = {}
        entity_dict = {}
    
    if cfg.backbones == "PNRLLM":
        with open(cfg.data_dir + '_train/refine_data_full.json', 'r') as f:
            enrichedE = json.load(f)

        with open(cfg.data_dir + '_train/genAbs2.json', 'r') as f:
            genAbs = json.load(f)
            listGenKey = list(genAbs.keys())

    category_dict = {}
    subcategory_dict = {}
    word_cnt = Counter()  # Counter is a subclass of the dictionary dict.

    num_line = len(open(file_path, encoding='utf-8').readlines())
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=num_line, desc=f"[{mode}]Processing raw news"):
            # split one line
            split_line = line.strip('\n').split('\t')
            news_id, category, subcategory, title, abstract, url, t_entity_str, _ = split_line
            if (cfg.backbones == "PNRLLM") and news_id in listGenKey:
                title = genAbs[news_id][0]
            update_dict(target_dict=news_dict, key=news_id)

            # Entity
            if t_entity_str:
                entity_ids = [obj["WikidataId"] for obj in json.loads(t_entity_str)]
                if (cfg.backbones == "PNRLLM") and (news_id in enrichedE):
                    tmp = [x[1] for x in enrichedE[news_id]]
                    currentid = 0
                    while (len(entity_ids) < cfg.entity_size) and (currentid < len(tmp) - 1):
                        if tmp[currentid] not in entity_ids:
                            entity_ids.append(tmp[currentid])
                        currentid += 1
                [update_dict(target_dict=entity_dict, key=entity_id) for entity_id in entity_ids]
            else:
                entity_ids = t_entity_str

            
            tokens = word_tokenize(title.lower())

            update_dict(target_dict=news, key=news_id, value=[tokens, category, subcategory, entity_ids,
                                                                news_dict[news_id]])

            if mode == 'train':
                update_dict(target_dict=category_dict, key=category)
                update_dict(target_dict=subcategory_dict, key=subcategory)
                word_cnt.update(tokens)

        if mode == 'train':
            word = [k for k, v in word_cnt.items() if v > 3]
            word_dict = {k: v for k, v in zip(word, range(1, len(word) + 1))}
            print(len(entity_dict))
            return news, news_dict, category_dict, subcategory_dict, entity_dict, word_dict
        else:  # val, test
            print(len(entity_dict))
            return news, news_dict, None, None, entity_dict, None

def prepare_preprocess_bin(cfg, mode):
    if cfg.reprocess is True:
        # Glove
        newsPath = os.path.join(cfg.data_dir + f'_{mode}', 'news.tsv')
        nltk_news, nltk_news_dict, category_dict, subcategory_dict, entity_dict, word_dict = read_raw_news(
            file_path=newsPath,
            cfg=cfg,
            mode=mode,
        )

        if mode == "train":
            pickle.dump(category_dict, open(os.path.join(cfg.data_dir + f'_{mode}', "category_dict.bin"), "wb"))
            pickle.dump(subcategory_dict, open(os.path.join(cfg.data_dir + f'_{mode}', "subcategory_dict.bin"), "wb"))
            pickle.dump(word_dict, open(os.path.join(cfg.data_dir + f'_{mode}', "word_dict.bin"), "wb"))
        else:
            category_dict = pickle.load(open(os.path.join(cfg.data_dir + '_train', "category_dict.bin"), "rb"))
            subcategory_dict = pickle.load(open(os.path.join(cfg.data_dir + '_train', "subcategory_dict.bin"), "rb"))
            word_dict = pickle.load(open(os.path.join(cfg.data_dir + '_train', "word_dict.bin"), "rb"))

        pickle.dump(entity_dict, open(os.path.join(cfg.data_dir + f'_{mode}', "entity_dict.bin"), "wb"))
        pickle.dump(nltk_news, open(os.path.join(cfg.data_dir + f'_{mode}', "nltk_news.bin"), "wb"))
        pickle.dump(nltk_news_dict, open(os.path.join(cfg.data_dir + f'_{mode}', "news_dict.bin"), "wb"))
        
        nltk_news_features = read_parsed_news(cfg, nltk_news, nltk_news_dict,
                                              category_dict, subcategory_dict, entity_dict,
                                              word_dict)
        news_input = np.concatenate([x for x in nltk_news_features], axis=1)
        pickle.dump(news_input, open(os.path.join(cfg.data_dir + f'_{mode}', "nltk_token_news.bin"), "wb"))
        print("Glove token preprocess finish.")
    else:
        print(f'[{mode}] All preprocessed files exist.')    


def prepare_preprocessed_data(cfg)  -> None:
    prepare_distributed_data(cfg, "train")
    prepare_distributed_data(cfg, "val")

    prepare_preprocess_bin(cfg, "train")
    prepare_preprocess_bin(cfg, "val")
    # prepare_preprocess_bin(cfg, "test")

    # # Entity vec process
    data_dir = {"train": cfg.data_dir + '_train', "val": cfg.data_dir + '_val', "test": cfg.data_dir}

    train_entity_emb_path = Path(data_dir['train']) / "entity_embedding.vec"
    val_entity_emb_path = Path(data_dir['val']) / "entity_embedding.vec"
    # test_entity_emb_path = Path(data_dir['test']) / "entity_embedding.vec"

    val_combined_path = Path(data_dir['val']) / "combined_entity_embedding.vec"
    # test_combined_path = Path(data_dir['test']) / "combined_entity_embedding.vec"

    # Open the first file and read the contents
    with open(train_entity_emb_path, 'r') as file1:
        file1_content = file1.readlines()
    # Open the second file and read the contents
    with open(val_entity_emb_path, 'r') as file2:
        file2_content = file2.readlines()

    merged_content = file1_content + file2_content

    # Write the concatenated content to the output file
    with open(val_combined_path, 'w') as output_file:
        output_file.writelines(merged_content)