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
import os
import glob
import torch
import ast
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader, IterableDataset
#from augment import AugmentModule
from g2pk import G2p

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
        if self.train_mode == 'denoise20':
            self.processor = G2p()


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
        if self.data_split_type == 'train' and self.train_mode == 'denoise20':
            tmp = source
            source = ' '.join([self.processor(x) if np.random.randint(10) in [0, 1] else x for x in source.split(" ")])
        input_ids, label_ids = self.tok.encode(source), self.tok.encode(target)

        input_ids = self.add_padding_data(input_ids)
        label_ids.append(self.tok.eos_token_id)
        dec_input_ids = [self.pad_index]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignored_data(label_ids)

#         return (torch.tensor(input_ids),
#                 torch.tensor(dec_input_ids),
#                 torch.tensor(label_ids))
        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_)}
    
    def __len__(self):
        return self.len
