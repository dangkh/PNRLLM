import logging
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
import random
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
class NewsCL_TrainDataset(Dataset):
	def __init__(self, filename, news_index, news_combined, cfg):
		super(NewsCL_TrainDataset).__init__()
		self.filename = filename
		self.news_index = news_index
		self.news_combined = news_combined
		self.user_log_length = cfg.his_size
		self.npratio = cfg.npratio
		self.cfg = cfg
		self.listPrep = []
		self.prepDatabyUser = []
		self.prepare()
		with open("listAllID.json", "w") as file:
			json.dump(self.listPrep, file)

	def trans_to_nindex(self, nids):
		return [self.news_index[i] if i in self.news_index else 0 for i in nids]

	def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
		if padding_front:
			pad_x = [padding_value] * (fix_length - len(x)) + x[-fix_length:]
			mask = [0] * (fix_length - len(x)) + [1] * min(fix_length, len(x))
		else:
			pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
			mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
		return pad_x, np.array(mask, dtype='float32')

	def prepare(self):
		self.preprocessDT = []
		with open(self.filename) as f:
			for line in tqdm(f):
				uid, dt = self.line_mapper(line)
				if len(uid) == 0:
					continue
					# remove those user has no history

				self.preprocessDT.append([uid,dt])
				if self.cfg.prototype and (len(self.preprocessDT) > 10000):
					break
	
	def line_mapper(self, line):
		line = line.strip().split('\t')
		uid = line[1]
		if uid not in self.listPrep:
			self.listPrep.append(uid)
			click_docs = line[3].split()
			originalClick = click_docs
			click_docs = self.trans_to_nindex(click_docs)
			numclick = len(click_docs)
			click_docs, log_mask = self.pad_to_fix_len(click_docs, self.user_log_length)
			if self.cfg.backbones == 'LSTUR':
				user_feature = [self.listPrep.index(uid), numclick, self.news_combined[click_docs]]
			else:
				user_feature = self.news_combined[click_docs]
			self.prepDatabyUser.append([originalClick, user_feature, log_mask])
		else:
			originalClick, _, _ = self.prepDatabyUser[self.listPrep.index(uid)]
		
		sess_pos = line[4].split()
		sess_neg = line[5].split()
		pos = self.trans_to_nindex(sess_pos)
		neg = self.trans_to_nindex(sess_neg)

		label = random.randint(0, self.npratio)
		sample_news = neg[:label] + pos + neg[label:]
		news_feature = self.news_combined[sample_news]

		return originalClick, [uid, torch.from_numpy(news_feature), torch.tensor(label)]


	def __getitem__(self, idx):
		_, [uid, news_feature, label] =  self.preprocessDT[idx]
		_, user_feature, log_mask = self.prepDatabyUser[self.listPrep.index(uid)]
		# if self.cfg.LSTUR:
		# 	meta1, meta2, actual_user_feature = user_feature
		# 	userIDX = torch.from_numpy(np.asarray([meta1, meta2]))
		return user_feature, torch.from_numpy(log_mask), news_feature, label

	def __len__(self):
		return len(self.preprocessDT)

class NewsCL_ValidDataset(NewsCL_TrainDataset):
	def __init__(self, filename, news_index, news_score, cfg):
		super(NewsCL_ValidDataset).__init__()
		self.filename = filename
		self.news_index = news_index
		self.news_score = news_score
		self.user_log_length = cfg.his_size
		self.npratio = cfg.npratio
		self.cfg = cfg
		self.listPrep = []
		self.listUser = []
		with open("listAllID.json", "r") as file:
			self.listUser = json.load(file)
		self.prepDatabyUser = []
		self.prepare()


	def line_mapper(self, line):
		line = line.strip().split('\t')
		uid = line[1]
		if uid not in self.listPrep:
			click_docs = line[3].split()
			originalClick = click_docs
			click_docs = self.trans_to_nindex(click_docs)
			numclick = len(click_docs)
			click_docs, log_mask = self.pad_to_fix_len(click_docs, self.user_log_length)
			if self.cfg.backbones == 'LSTUR':
				suid = 50000
				if uid in self.listUser:
					suid = self.listUser.index(uid)
				user_feature = [suid, numclick, self.news_score[click_docs]]
			else:
				user_feature = self.news_score[click_docs]

			self.prepDatabyUser.append([originalClick, user_feature, log_mask])
			self.listPrep.append(uid)
		else:
			originalClick, _, _ = self.prepDatabyUser[self.listPrep.index(uid)]

		candidate_news = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
		label = np.array([int(i.split('-')[1]) for i in line[4].split()])
		news_feature = self.news_score[candidate_news]

		return originalClick, [uid, torch.from_numpy(news_feature), torch.tensor(label)]



class NewsDataset(Dataset):
	def __init__(self, data):
		self.data = data

	def __getitem__(self, idx):
		return self.data[idx]

	def __len__(self):
		return self.data.shape[0]

def load_dataloader(cfg, mode='train', model=None):
	data_dir = {"train": cfg.data_dir + '_train', "val": cfg.data_dir + '_val', "test": cfg.data_dir + '_val'}

	# ------------- load news.tsv-------------
	news_index = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))

	news_input = pickle.load(open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb"))
	# ------------- load behaviors_np{X}.tsv --------------
	if mode == 'train':
		target_file = Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_0.tsv"

		dataset = NewsCL_TrainDataset(
			filename=target_file,
			news_index=news_index,
			news_combined=news_input,
			cfg=cfg
		)
		dataloader = DataLoader(dataset, batch_size=cfg.batch_size)
	elif mode in ['val', 'test']:
		# convert the news to embeddings
		news_dataset = NewsDataset(news_input)
		news_dataloader = DataLoader(news_dataset,  batch_size= 128)

		news_scoring = []
		with torch.no_grad():
			for input_ids in tqdm(news_dataloader):
				if cfg.backbones == "PNRLLM":
					e_lis = input_ids[:,-5:]
					input_ids = input_ids[:,:-5].cuda()
				else:
					input_ids = input_ids.cuda()
				news_vec = model.news_encoder(input_ids)
				news_vec = news_vec.to(torch.device("cpu"))
				if cfg.backbones == "PNRLLM":
					news_vec = torch.concatenate((news_vec, e_lis),-1)
				news_vec = news_vec.detach().numpy()
				news_scoring.extend(news_vec)

		news_scoring = np.array(news_scoring)

		
		if mode == 'val':
			dataset = NewsCL_ValidDataset(
				filename=Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_0.tsv",
				news_index=news_index,
				news_score=news_scoring,
				cfg=cfg
			)

			dataloader = DataLoader(dataset, batch_size=1)

		else:
			dataset = NewsCL_ValidDataset(
					filename=Path(data_dir[mode]) / f"behaviors.tsv",
					news_index=news_index,
					news_score=news_scoring,
					cfg=cfg
				)

			dataloader = DataLoader(dataset, batch_size=1)
		

	return dataloader
