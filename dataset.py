import pytorch_lightning as pl
import os
import glob
import torch
import ast
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader, IterableDataset

class GecDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer, DatasetModule):
        super().__init__()                                                        
        self.args = args                       
        self.tokenizer = tokenizer                            
        self.train = DatasetModule(self.args.train_data_path, self.tokenizer,
                self.args.max_seq_len, data_split_type='train')
        self.valid = DatasetModule(self.args.val_data_path, self.tokenizer,
                self.args.max_seq_len, data_split_type='test')
        self.test = DatasetModule(self.args.test_data_path, self.tokenizer,           
                self.args.max_seq_len, data_split_type='test')  

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

    def train_noshuffle_dataloader(self):
        return DataLoader(self.train, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)


class KoBARTGecDataset(Dataset):
    def __init__(self, filename, tok, max_len, pad_index = 0, ignore_index=-100, data_split_type='val', train_mode='normal'):
        super().__init__()
        self.tok = tok
        self.max_len = max_len
        self.docs = self.read_docs(filename)
        self.len = len(self.docs)
        self.pad_index = pad_index
        self.ignore_index = ignore_index
        self.train_mode = train_mode
        self.data_split_type = data_split_type

    def read_docs(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data = f.read().split("\n")
        # process - leave out null columns and invalid texts
        data = [x.split('\t') for x in data if x != '']
        data = [x for x in data if len(x) == 2 and x[0] != '' and x[1] != '']
        return data


    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def __getitem__(self, idx):
        instance = self.docs[idx]
        source, target = instance
        input_ids, label_ids = self.tok.encode(source), self.tok.encode(target)

        input_ids = self.add_padding_data(input_ids)
        label_ids.append(self.tok.eos_token_id)
        dec_input_ids = [self.pad_index]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignored_data(label_ids)

        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_)}

    def __len__(self):
        return self.len
