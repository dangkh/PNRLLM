import argparse
import numpy as np
from torch import nn
from src.config import TrainConfig 
from src.ultis import *
from src.data_helper import prepare_preprocessed_data
from src.data_load import *
from src.metrics import *
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def acc(y_true, y_hat):
	y_hat = torch.argmax(y_hat, dim=-1)
	tot = y_true.shape[0]
	hit = torch.sum(y_true == y_hat)
	return hit.data.float() * 1.0 / tot

def get_mean(arr):
	return [np.array(i).mean() for i in arr]

def evaluate_modelPanel(model, cfg, mode = 'val'):

	model.eval()
	with torch.no_grad():
		valid_dataloader = load_dataloader(cfg, mode, model)
		tasks = []
		AUC = []
		MRR = []
		nDCG5 = []
		nDCG10 = []
		for cnt, (log_vecs, log_mask, news_vecs, labels) in tqdm(enumerate(valid_dataloader)):
			log_vecs = log_vecs.cuda()
			log_mask = log_mask.cuda()

			if cfg.backbones == "PNRLLM":
				news_vecs = news_vecs.cuda()
				e_his = log_vecs[:,:,-5:].int()
				log_vecs = log_vecs[:,:,:-5]
				e_candi = news_vecs[:,:,-5:].int()
				news_vecs = news_vecs[:,:,:-5]

				e_his = model.entity_embedding_layer(e_his)
				e_candi = model.entity_embedding_layer(e_candi)
				e_his = model.entity_encoder(e_his, None)
				e_candi = model.entity_encoder(e_candi, None)
			
				user_vecs = model.user_att(torch.stack([log_vecs, e_his], dim=2).view(-1, 2, model.news_dim))
				user_vecs = user_vecs.view(-1, model.user_log_length, model.news_dim)
				news_vecs = model.candi_att(torch.stack([news_vecs, e_candi], dim=2).view(-1, 2, model.news_dim))
				news_vecs = news_vecs.unsqueeze(0).detach().cpu().numpy()

			else:
				user_vecs = log_vecs.view(-1, model.user_log_length, model.news_dim)

			user_vecs = model.user_encoder(user_vecs, log_mask)

			user_vecs = user_vecs.detach().cpu().numpy()
			
			labels = labels.detach().cpu().numpy()

			for user_vec, news_vec, label in zip(user_vecs, news_vecs, labels):
				tmp = np.mean(label)
				if tmp == 0 or tmp == 1:
					continue

				score = np.dot(news_vec, user_vec)
				auc = roc_auc_score(label, score)
				mrr = mrr_score(label, score)
				ndcg5 = ndcg_score(label, score, k=5)
				ndcg10 = ndcg_score(label, score, k=10)

				AUC.append(auc)
				MRR.append(mrr)
				nDCG5.append(ndcg5)
				nDCG10.append(ndcg10)
	reduced_auc, reduced_mrr, reduced_ndcg5, reduced_ndcg10 = get_mean([AUC, MRR, nDCG5, nDCG10])
	res = {
		"auc": reduced_auc,
		"mrr": reduced_mrr,
		"ndcg5": reduced_ndcg5,
		"ndcg10": reduced_ndcg10,
	}
	
	return res

def train_modelPanel(model, optimizer, dataloader, cfg):
	model = model.to(device)
	model.train()
	torch.set_grad_enabled(True)
	for ep in range(cfg.epochs):
		loss = 0.0
		accuary = 0.0
		print("EPOCH: " + str(ep))
		for cnt, (user_feature, log_mask, news_feature, targets) in tqdm(enumerate(dataloader)):
			user_feature = user_feature.to(device)
			log_mask = log_mask.to(device)
			news_feature = news_feature.to(device)
			targets = targets.to(device)

			
			bz_loss, y_hat = model([user_feature, log_mask, news_feature, targets])
			bz_loss = bz_loss.mean()  # chỉ dùng mẫu được chọn

			acc_val = acc(targets, y_hat)

			loss += bz_loss.item()
			accuary += acc_val

			optimizer.zero_grad()
			bz_loss.backward()
			optimizer.step()

		# Update lambda
		print(loss, accuary)
		
		checkpoint = {
			'epoch': cfg.epochs,  
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': loss,  # Save loss or any other metric
		}
		# Save the checkpoint
		torch.save(checkpoint, f'./checkpoint/{ep}_use_model_{cfg.backbones}.pth')
		logging.info("Evaluation")
		testRes = evaluate_modelPanel(model, cfg, mode='test')
		print(testRes)
		model.train()

cfg = TrainConfig()

logging.info("Start")
print(cfg)
set_random_seed(cfg.random_seed)
"""
0. Definite Parameters & Functions

"""
logging.info("Prepare the dataset")
# if using Enriched Entity, make sure that reprocess setting is True
prepare_preprocessed_data(cfg)
train_dataloader = load_dataloader(cfg, mode='train')

logging.info("Initialize Model")
model, optimizer = load_model(cfg)
print(model)
logging.info("Training Start")
train_modelPanel(model, optimizer, train_dataloader, cfg)
logging.info("End")