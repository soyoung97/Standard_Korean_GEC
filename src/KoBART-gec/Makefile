CUDANUM = 0
SEED = 0
LR = 3e-05
LR3 = 3e-05
PRETRAINED_PATH = '' #'logs/wiki_pretrained/kobart_gec-model_chp/wiki_2e-05_epoch=03.ckpt'
DIR = '../../output'
EPOCHS = 10
CUSTOM = $(NOISEDLANG8)
MODE = normal # normal
DATA = korean_learner

KL = ../../extract_data/korean_learner/korean_learner
UNION = ../extract_data/union/union
NATIVE = ../../extract_data/native/native
LANG8 = ../../extract_data/lang8/lang8
WIKI = ../../extract_data/wiki/wiki
UNION = ../../extract_data/union/union
NOISEDKL = ../../extract_data/noised_korean_learner/noised_korean_learner
NOISEDNATIVE = ../../extract_data/noised_native/noised_native
NOISEDLANG8 = ../../extract_data/noised_lang8/noised_lang8



custom:
	CUDA_VISIBLE_DEVICES=$(CUDANUM) python3 train.py --train_mode $(MODE) --data $(DATA) --default_root_dir $(DIR) --max_epochs $(EPOCHS) --lr $(LR) --from_pretrained $(PRETRAINED_PATH) --SEED $(SEED)

kl:
	CUDA_VISIBLE_DEVICES=$(CUDANUM) python3 train.py --train_mode $(MODE) --data korean_learner --default_root_dir $(DIR) --max_epochs $(EPOCHS) --lr $(LR) --from_pretrained $(PRETRAINED_PATH) --SEED $(SEED)

native:
	CUDA_VISIBLE_DEVICES=$(CUDANUM) python3 train.py --train_mode $(MODE) --data native --default_root_dir $(DIR) --max_epochs $(EPOCHS) --lr $(LR) --from_pretrained $(PRETRAINED_PATH) --SEED $(SEED)

lang8:
	CUDA_VISIBLE_DEVICES=$(CUDANUM) python3 train.py --train_mode $(MODE) --data lang8 --default_root_dir $(DIR) --max_epochs $(EPOCHS) --lr $(LR) --from_pretrained $(PRETRAINED_PATH) --SEED $(SEED)

wiki:
	CUDA_VISIBLE_DEVICES=$(CUDANUM) python3 train.py --train_mode $(MODE) --data wiki --default_root_dir $(DIR) --max_epochs $(EPOCHS) --lr $(LR) --from_pretrained $(PRETRAINED_PATH) --SEED $(SEED)

union:
	CUDA_VISIBLE_DEVICES=$(CUDANUM) python3 train.py --train_mode $(MODE) --data union --default_root_dir $(DIR) --max_epochs $(EPOCHS) --lr $(LR) --from_pretrained $(PRETRAINED_PATH) --SEED $(SEED)

wiki_pretrain:
	CUDA_VISIBLE_DEVICES=3 python3 train.py --default_root_dir logs/wiki_pretrained --max_epochs 30 --lr 2e-05 --from_pretrained $(PRETRAINED_PATH) --SEED $(SEED)

noisedkl:
	CUDA_VISIBLE_DEVICES=$(CUDANUM) python3 train.py --train_mode $(MODE) --data korean_learner --default_root_dir logs/noisedkl --max_epochs $(EPOCHS) --lr $(LR) --from_pretrained $(PRETRAINED_PATH) --SEED $(SEED)

noisednative:
	CUDA_VISIBLE_DEVICES=$(CUDANUM) python3 train.py  --train_mode $(MODE) --data native --default_root_dir logs/noisednative --max_epochs $(EPOCHS) --lr $(LR) --from_pretrained $(PRETRAINED_PATH) --SEED $(SEED)

noisedlang8:
	CUDA_VISIBLE_DEVICES=$(CUDANUM) python3 train.py --train_mode $(MODE) --data lang8  --default_root_dir logs/noisedlang8 --max_epochs $(EPOCHS) --lr $(LR) --from_pretrained $(PRETRAINED_PATH) --SEED $(SEED)







