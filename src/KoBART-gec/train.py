import argparse
import logging
from pathlib import Path
import os
import re
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset
from dataset import KoBARTGecDataset
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
from metric.gleumodule import run_gleu
from transformers import set_seed
import savvihub
from pprint import pprint
import time

parser = argparse.ArgumentParser(description='KoBART gec')

parser.add_argument('--checkpoint_path',
                    type=str,
                    help='checkpoint path')
parser.add_argument('--train_mode',
                    type=str,
                    default='normal',
                    help='dataset denoising (training) mode - can select between normal and denoise10')
parser.add_argument('--data',
                    type=str,
                    default='korean_learner',
                    help='(fine-tuning) dataset name: to be used as reference to search for m2 files')

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size',
                            type=int,
                            default=32,
                            help='Batch Size for Training')

        parser.add_argument('--max_len',
                            type=int,
                            default=128,
                            help='max seq len')
        parser.add_argument('--SEED',
                            type=int,
                            default=0,
                            help='seed for the model to train')
        parser.add_argument('--from_pretrained',
                            type=str,
                            default='',
                            help='filename inside the mdoel log directory if we are to load model from pretrained')
        parser.add_argument('--dropout',
                            type=float,
                            default=0.1,
                            help="dropout for KoBART config")
        return parser

class KobartGecModule(pl.LightningDataModule):
    def __init__(self, train_file, valid_file,
                 test_file,
                 max_len,
                 batch_size=32,
                 num_workers=5, train_mode='normal'):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_file_path = train_file
        self.valid_file_path = valid_file
        self.test_file_path = test_file
        self.tok = get_kobart_tokenizer()
        self.num_workers = num_workers
        self.train_mode = train_mode

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',
                            type=int,
                            default=5,
                            help='num of worker for dataloader')
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.train = KoBARTGecDataset(self.train_file_path,
                                 self.tok,
                                 self.max_len, data_split_type='train', train_mode=self.train_mode)
        self.test = KoBARTGecDataset(self.test_file_path,
                                self.tok,
                                self.max_len, data_split_type='test', train_mode=self.train_mode)
        self.valid = KoBARTGecDataset(self.valid_file_path,
                                self.tok,
                                self.max_len, data_split_type='test', train_mode=self.train_mode)

    def train_dataloader(self):
        train = DataLoader(self.train,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.valid,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)
        return test


class Base(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super(Base, self).__init__()
        self.hparams = hparams

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--lr',
                            type=float,
                            default=3e-5,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')

        parser.add_argument('--model_path',
                            type=str,
                            default=None,
                            help='kobart model path')
        return parser

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_workers = (self.hparams.gpus if self.hparams.gpus is not None else 1) * (self.hparams.num_nodes if self.hparams.num_nodes is not None else 1)
        data_len = len(self.train_dataloader().dataset)
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]


class KoBARTConditionalGeneration(Base):
    def __init__(self, hparams, **kwargs):
        super(KoBARTConditionalGeneration, self).__init__(hparams, **kwargs)
        self.args = hparams
        config = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model()).config
        config.dropout = self.args.dropout
        self.model = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model(), config=config)
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.pad_token_id = 0
        self.tokenizer = get_kobart_tokenizer()
        self.epoch = 0
        self.outputs = []
        self.decoded_labels = []
        self.origs = []
        self.step = 0
        self.max_len = self.args.max_len
        self.scores = {}
        self.generation_time = 0

    def forward(self, inputs):
        attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float()
        return self.model(input_ids=inputs['input_ids'],
                attention_mask=attention_mask,
                decoder_input_ids=inputs['decoder_input_ids'],
                decoder_attention_mask=decoder_attention_mask,
                labels=inputs['labels'], return_dict=True)

    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=False)
        #savvihub.log(step=self.step, row={"training_loss": loss.item()})
        self.step += 1
        return loss

    def training_epoch_end(self, _):
        savvihub.log(step=self.epoch, row={"generation-time": self.generation_time})
        self.scores[self.epoch]['generation_time'] = self.generation_time
        print(f"\nGeneration time: {self.generation_time}")
        self.generation_time = 0
        self.epoch += 1

    def generate(self, input_ids, labels):
        self.model.eval()
        start = time.time()
        output = self.model.generate(input_ids, eos_token_id=1, max_length=self.max_len, num_beams=4)
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        end = time.time()
        self.generation_time += end - start
        decoded_label = self.tokenizer.batch_decode(labels.masked_fill(labels == -100, 1), skip_special_tokens=True)
        self.outputs += output
        self.decoded_labels += decoded_label
        self.origs += self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    def should_generate(self):
        return True

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        #savvihub.log(step=self.step, row={"validation_loss": loss.item()})
        if self.should_generate():
            self.generate(batch['input_ids'], batch['labels'])
        return (loss)

    def test_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        #savvihub.log(step=self.step, row={"test_loss": loss.item()})
        if self.should_generate():
            self.generate(batch['input_ids'], batch['labels'])
        return (loss)
    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, mode='test')

    def validation_epoch_end(self, outputs, mode='val'):
        losses = []
        for loss in outputs:
            losses.append(loss)
        total_loss = torch.stack(losses).mean()
        if self.should_generate():
            # Make generation output directory and file
            directory = f"{self.args.default_root_dir}/generation/epoch{self.epoch}/{mode}"
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            with open(directory + f"/hypothesis_{total_loss}.txt", 'w', encoding='utf-8') as f:
                f.write("\n".join(self.outputs))
            with open(directory + f"/reference_{total_loss}.txt", "w", encoding='utf-8') as f:
                f.write("\n".join(self.decoded_labels))
            with open(directory + f"/source_{total_loss}.txt", "w", encoding='utf-8') as f:
                f.write("\n".join(self.origs))
            
            self.outputs = []
            self.decoded_labels = []
            self.origs = []
            gleu_out = run_gleu(reference=directory + f"/reference_{total_loss}.txt", source=directory + f"/source_{total_loss}.txt", hypothesis=directory + f"/hypothesis_{total_loss}.txt")
            logging.info(f"\ngleu_value: {gleu_out}\n")
            with open(directory + f"/gleu_{gleu_out}.txt", "w", encoding='utf-8') as f:
                f.write(f"data: {self.args.data}, epoch: {self.epoch}, gleu_out: {gleu_out}, val_loss: {total_loss}\nhparams:{self.hparams}")
            #self.split_punct(f"{directory}/hypothesis_{total_loss}.txt")
            # output m2scores
            system_command = f"./metric/m2scorer/m2scorer {directory}/hypothesis_{total_loss}.txt ../../extract_data/{self.args.data}/{self.args.data}_{mode}.m2 > {directory}/m2score.txt"
            print(f"Processing: {system_command}")
            out = os.system(system_command)
            if out != 0:
                print("Trying to run m2scorer from parent path")
                system_command = f"./src/KoBART-gec/metric/m2scorer/m2scorer {directory}/hypothesis_{total_loss}.txt extract_data/{self.args.data}/{self.args.data}_{mode}.m2 > {directory}/m2score.txt"
                print("Retrying: ", system_command)
                out = os.system(system_command)
            os.system(f"cat {directory}/m2score.txt")
            with open(f"{directory}/m2score.txt", encoding='utf-8') as f:
                try:
                    p, r, f_score = [x.split(": ")[1].strip() for x in f.read().strip().split("\n")]
                except:
                    print("Reading m2 scores failed")
                    p, r, f_score = 0, 0, 0
        gleuscore = float(gleu_out) * 100
        self.scores[self.epoch] = {'precision': p, 'recall': r, 'f_score': f_score, 'gleu': gleuscore, 'loss': total_loss.item()}
        print(f"\n\nEPOCH {self.epoch} / VAL_LOSS {round(total_loss.item(), 2)} / GLEU {round(gleuscore, 2)}\n\n")
        savvihub.log(step=self.epoch, row={f"{mode}_glue_score": round(gleuscore, 2), f"total_{mode}_loss": total_loss.item(), f"{mode}_m2scoreP": p, f"{mode}_m2scoreR": r, 
        f"{mode}_m2scoreF": f_score})
        self.log(f'{mode}_loss', total_loss, prog_bar=False)
        self.log(f'{mode}_gleu', gleuscore)
        if self.args.best['gleu'] < gleuscore:
            self.args.best['gleu'] = gleuscore
            self.args.best['f0.5'] = f_score
            self.args.best['prec'] = p
            self.args.best['rec'] = r
        print(f"Print ordering of: precision, recall, f0.5 score, gleu score, val loss, generation_time. The last epoch is test epoch.")
        pprint(self.scores)

def configure_fixed_args(args):
    #args.default_root_dir = 'logs'
    args.gpus = 1
    args.gradient_clip_val = 1.0
    args.max_len = 128
    data = args.data
    if os.path.exists(f"../../extract_data/{data}/{data}_train.txt"):
        data_dir = "../../extract_data/"
    elif os.path.exists(f"extract_data/{data}/{data}_train.txt"):
        data_dir = "extract_data/"
    else:
        raise Exception("Wrong dataname or file path: file does not exist")
    args.train_file = f"{data_dir}/{data}/{data}_train.txt"
    args.valid_file = f"{data_dir}/{data}/{data}_val.txt"
    args.test_file = f"{data_dir}/{data}/{data}_test.txt"
    set_seed(args.SEED)
    return args

if __name__ == '__main__':
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = KobartGecModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args = configure_fixed_args(args)
    args.best = {'gleu': 0, 'prec': 0, 'rec': 0, 'f0.5': 0}
    logging.info(args)
    model = KoBARTConditionalGeneration(args)
    if args.from_pretrained != '':
        print(f"Loading from pretrained model: {args.from_pretrained}")
        model = KoBARTConditionalGeneration.load_from_checkpoint(checkpoint_path=f"{args.from_pretrained}", hparams=args)
    dm = KobartGecModule(args.train_file, args.valid_file,
                        args.test_file,
                        args.max_len,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers, train_mode=args.train_mode)
    LR = args.lr
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_gleu',
                                                       dirpath=args.default_root_dir,
                                                       filename=f'model_chp/{args.data}_{LR}_'+'{epoch:02d}',
                                                       verbose=True,
                                                       save_last=False,
                                                       mode='max',
                                                       save_top_k=1,
                                                       prefix='kobart_gec')
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger,
                                            callbacks=[checkpoint_callback, lr_logger])
    trainer.fit(model, dm)
    trainer.test(model)
    for _ in range(10):
        print(args.best)
