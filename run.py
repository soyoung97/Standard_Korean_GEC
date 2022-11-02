'''
https://github.com/soyoung97/Standard_Korean_GEC
Modified MIT License

Software Copyright (c) 2022 Soyoung Yoon

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
The above copyright notice and this permission notice need not be included
with content created by the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
'''
import multiprocessing
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from dataset import KoBARTGecDataset, GecDataModule
from model import KoBARTConditionalGeneration
from pprint import pprint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import datetime
import argparse
from pytz import timezone


def parse_common_args():
    parser = argparse.ArgumentParser(description='Common arguments')
    parser.add_argument('--data', type=str, default='mtop', help='Type of dataset to train')
    parser.add_argument('--max_epochs', type=int, default=100, help='Total number of epochs to train')
    parser.add_argument('--debug', action='store_true', help='If true, reduces the number of validation & test dataset for faster loading and debugging')

    # mode-specific arguments

    parser.add_argument('--name', type=str, default='default_name', help='Name tag, just like memo, that is included in the output file save')
    parser.add_argument('--max_seq_len', type=int, default=128, help='maximum token length for the tokenizer')
    parser.add_argument('--seed', type=int, default=0, help='seed to train model and split data')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-05, help='Set learning rate for optimizer')
    # dataset arguments
    parser.add_argument('--train_data_path', type=str, default='data/wikisql/1000/wikisql_train.jsonl', help='wikisql train data path')
    parser.add_argument('--val_data_path', type=str, default='data/wikisql/1000/wikisql_val.jsonl', help='wikisql validation data path')
    parser.add_argument('--test_data_path', type=str, default='data/wikisql/1000/wikisql_test.jsonl', help='wikisql test data path')
    # model checkpoint arguments
    parser.add_argument('--every_n_epochs', type=int, default=10, help='Save model checkpoint every n epochs')
    parser.add_argument("--model_ckpt_path", type=str, default='', help='Path to load model checkpoint, empty string in default')
    # logging settings
    parser.add_argument('--log_every_n_steps', type=int, default=50, help='do logging at every n steps.')
    # eval settings
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='validate at every n epochs.')
    # training settings
    parser.add_argument('--warmup_ratio', type=float, default=0.0, help='warmup steps ratio')
    return parser

def current_time():
    current_time = datetime.datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d_%H-%M-%S')
    return current_time

def write_command_logs():
    command_line = ' '.join(sys.argv)
    command_line = "python3 " + command_line
    cur_time = current_time()
    try:
        devices = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        devices = ','.join([str(x) for x in list(range(torch.cuda.device_count()))])
    with open("logs/command_logs.txt", 'a') as f:
        f.write(f"[{cur_time}]: CUDA_VISIBLE_DEVICES={devices} {command_line}\n")


def cli_main():
    write_command_logs()
    parser = parse_common_args()
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.num_sanity_val_steps = 0
    if args.debug:
        args.num_workers = 0
        args.batch_size = 16
    else:
        args.num_workers = int(multiprocessing.cpu_count()/2)
    args.best = {'gleu': 0, 'prec': 0, 'rec': 0, 'f0.5': 0}
    pl.seed_everything(args.seed)
    print("Arguments: ")
    pprint(vars(args))
    run_mode(args)

def get_device_count():
    if not torch.cuda.is_available():
        return 0
    try:
        devices = len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))
    except KeyError:
        devices = torch.cuda.device_count()
    return devices

def run_mode(args):

    # ------------
    # model & tokenizer setup
    # ------------

    config = BartForConditionalGeneration.from_pretrained('hyunwoongko/kobart').config
    bart_model = BartForConditionalGeneration.from_pretrained('hyunwoongko/kobart', config=config)
    tokenizer = PreTrainedTokenizerFast.from_pretrained('hyunwoongko/kobart')
    dm = GecDataModule(args, tokenizer, KoBARTGecDataset)
    model = KoBARTConditionalGeneration(args, bart_model, tokenizer, dm)
   
    # ------------
    # Defining Callbacks
    # ------------

    ckpt_callback = ModelCheckpoint(
        monitor='val_gleu',
        dirpath=f'outputs/{args.data}/',
        mode='max',
        verbose=True,
        save_last=False,
        save_top_k=-1,
        every_n_epochs=args.every_n_epochs,
        filename=f'model_ckpt/{args.data}_{args.lr}_' + '{epoch:02d}'
        )
    # make sure log steps are smaller than step per train
    data_len = len(dm.train_dataloader().dataset)
    args.log_every_n_steps = max(min(int(data_len / (args.batch_size*2)), args.log_every_n_steps), 1)
    print(f"Adjusted args:")
    pprint(vars(args))
    # ------------
    # Calling Trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(
        args, 
        accelerator='gpu',
        callbacks=[ckpt_callback], 
        devices=get_device_count(), 
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        strategy='dp')

    if args.model_ckpt_path == '':
        trainer.fit(model, dm)
    else:
        print(f"Loading model from {args.model_ckpt_path}...")
        model = model.load_from_checkpoint(checkpoint_path=args.model_ckpt_path, args=args, model=bart_model, tokenizer=tokenizer, datamodules=dm)
        trainer.validate(model, dataloaders=dm)
    # ------------
    # testing
    # ------------
    #result = trainer.test(dataloaders=dm.test_dataloader())


if __name__ == '__main__':
    cli_main()
