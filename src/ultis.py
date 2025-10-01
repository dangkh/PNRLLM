import os
import random
import numpy as np
import torch
from typing import Callable
import logging
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score
import src.models.backbone as bb
import pickle
from pathlib import Path
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def set_random_seed(random_seed: int = 42) -> None:
	random.seed(random_seed)
	os.environ["PYTHONHASHSEED"] = str(random_seed)
	np.random.seed(random_seed)
	torch.manual_seed(random_seed)
	torch.cuda.manual_seed(random_seed)
	torch.backends.cudnn.benchmark = True


def load_pretrain_emb(embedding_file_path, target_dict, target_dim):
	embedding_matrix = np.random.rand(len(target_dict) + 1, target_dim)
	# embedding_matrix = np.zeros(shape=(len(target_dict) + 1, target_dim))
	have_item = []
	if embedding_file_path is not None:
		with open(embedding_file_path, 'rb') as f:
			while True:
				line = f.readline()
				if len(line) == 0:
					break
				line = line.split()
				itme = line[0].decode()
				if itme in target_dict:
					index = target_dict[itme]
					tp = [float(x) for x in line[1:]]
					embedding_matrix[index] = np.array(tp)
					have_item.append(itme)
	have_item = set(have_item)
	print('-----------------------------------------------------')
	print(f'Dict length: {len(target_dict)}')
	print(f'Have words: {len(have_item)}')
	miss_rate = (len(target_dict) - len(have_item)) / len(target_dict) if len(target_dict) != 0 else 0
	print(f'Missing rate: {miss_rate}')
	return embedding_matrix


def load_model(cfg):
	category_dict = pickle.load(open(os.path.join(cfg.data_dir + '_train', "category_dict.bin"), "rb"))
	subcategory_dict = pickle.load(open(os.path.join(cfg.data_dir + '_train', "subcategory_dict.bin"), "rb"))
	if cfg.backbones == "PNRLLM":
		entity_dict = pickle.load(open(Path(cfg.data_dir + '_val') / "entity_dict.bin", "rb"))
		entity_emb_path = Path(cfg.data_dir + '_val') / "combined_entity_embedding.vec"
		entity_emb = load_pretrain_emb(entity_emb_path, entity_dict, 100)
	else:
		entity_emb = None
	word_dict = pickle.load(open(os.path.join(cfg.data_dir + '_train', "word_dict.bin"), "rb"))
	glove_emb = load_pretrain_emb(cfg.glove_path, word_dict, cfg.word_emb_dim)
	loadModel = getattr(bb, cfg.backbones)
	model = loadModel(glove_emb, entity_emb, len(category_dict), len(subcategory_dict), cfg)
	optimizer = optim.Adam(model.parameters(), lr=0.0003)

	return model, optimizer

def save_model(cfg, model, optimizer=None, mark=None):
	file_path = Path(f"{cfg.model_name}.pth")
	file_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
		},
		file_path)
	print(f"Model Saved. Path = {file_path}")


class EarlyStopping:
	"""
	Early Stopping class
	"""

	def __init__(self, patience=3):
		self.patience = patience
		self.counter = 0
		self.best_score = 0.0

	def __call__(self, score):
		"""
		The greater score, the better result. Be careful the symbol.
		"""
		if score > self.best_score:
			early_stop = False
			get_better = True
			self.counter = 0
			self.best_score = score
		else:
			get_better = False
			self.counter += 1
			if self.counter >= self.patience:
				early_stop = True
			else:
				early_stop = False

		return early_stop, get_better