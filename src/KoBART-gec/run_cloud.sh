#pip install git+https://github.com/SKT-AI/KoBART#egg=kobart
#pip install transformers==4.0.0
#pip install pytorch-lightning==1.1.0
#pip install pandas
#pip install torchtext==0.8.1
#pip install torch==1.7.1
python src/KoBART-gec/train.py --gradient_clip_val 1.0 --lr 1e-06 --max_epochs 50 --default_root_dir logs --gpus 1 --batch_size 4 --train_file src/KoBART-gec/data/korean_learner_train.txt --test_file src/KoBART-gec/data/korean_learner_val.txt
