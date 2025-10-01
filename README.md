# PNRLLM


##  Requirements:

```
torch, pydantic, 
```
##  Usage



### Training:

```


python train.py 


```
#### Configs in src/config.py:
```
random_seed: int = 1009
npratio: int = 4
history_size: int = 50
batch_size: int = 128
gradient_accumulation_steps: int = 8  # batch_size = 16 x 8 = 128
epochs: int = 3
learning_rate: float = 3e-4
weight_decay: float = 1e-5
max_len: int = 100
reprocess: bool = True
data_dir: str = "./data/MINDsmall"
gpu_num: int = 1
title_size: int = 50
abstract_size: int = 50
entity_size: int = 5
model_name: str = "NewsCL"
glove_path: str = './data/glove.840B.300d.txt'
word_emb_dim: int = 300
head_num: int = 4
head_dim: int = 100
entity_emb_dim: int = 100
attention_hidden_dim: int = 200
dropout_probability: float = 0.2
his_size: int = 50
early_stop_patience: int = 5
prototype: bool  = False
backbones: str = 'PNRLLM'# [PNRLLM, NAML, LSTUR, NRMS]
```

#### Results:
| Models      | AUC        | MRR          | N@5          | N@10         |
| :----:      |    :----:  |    :----:    |    :----:    |    :----:    |
| NewsCL+PNRLLM     
| NewsCL+NAML          
| NewsCL+NRMS        
| NewsCL+LSTUR         


## Dataset:
```
MIND-SMALL, ADRESSA
Link LLMs generated data: TBD
```

```
NewsCL
│
├── data/
│   ├── MINDsmall_train/
│   │   ├── behaviors.tsv
│   │   ├── news.tsv
│   │   ├── entity_embedding.vec
│   │   └── relation_embedding.vec
│	├── MINDsmall_train/
│   │   ├── behaviors.tsv
│   │   ├── news.tsv
│   │   ├── entity_embedding.vec
│   │   └── relation_embedding.vec
└───└ glove.840B.300d.txt
```

