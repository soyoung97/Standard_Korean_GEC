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
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
import numpy as np


class AugmentModule():
    def __init__(self, seed=0, mask_ratio=0.1, random_ratio=0.1, insert_ratio=0.1, rotate_ratio=0.1, permute_sentence_ratio=0.1):
        self.seed = seed
        self.tok = get_kobart_tokenizer()
        self.mask_ratio = mask_ratio
        self.random_ratio = random_ratio
        self.insert_ratio = insert_ratio
        self.rotate_ratio = rotate_ratio
        self.permute_sentence_ratio = permute_sentence_ratio

    def augment(self, input_ids, label_ids):
        return input_ids, label_ids
        # TODO: set same function of: with data_utils.numpy_seed(self.seed, self.epoch, index) for np.random.random()
        if self.permute_sentence_ratio > 0:
            input_ids = self.permute_sentences(input_ids)
        if self.mask_ratio > 0:
            input_ids = self.add_whole_word_mask(input_ids)
        if self.insert_ratio > 0:
            input_ids = self.add_insertion_noise(input_ids)

        if self.rotate_ratio > 0 and np.random.random() < self.rotate_ratio:
            input_ids = self.add_rolling_noise(input_ids)

        input_ids, label_ids = self.transform_func(input_ids, label_ids)
	return input_ids, label_ids

    def permute_sentences(self, input_ids): # TODO: implement this in input_ids way?
	source = self.tok.decode(input_ids)
        split_sent = source.split(".")
        sentence_idxs = [idx for idx, s in enumerate(split_sent) if len(s) != 0]
        if len(sentence_idxs) == 1:
            return source
        output = list(split_sent)
        mixed_idxs = np.random.permutation(sentence_idxs)
        for orig_idx, mixed_idx in zip(sentence_idxs, mixed_idxs):
            output[orig_idx] = split_sent[mixed_idx]
        return tok.encode(".".join(output))

    def add_whole_word_mask(self, input_ids):
        num_to_mask = int(len(input_ids) * 0.1) # TODO: subject to change
        indices = range(1, len(input_ids))
        indices_to_mask = np.array(random.sample(indices, num_to_mask))
        input_ids[incdices_to_mask] = 6 # 6 is <mask>
        return self.tok.decode(input_ids)

    def add_insertion_noise(self, input_ids): #TODO: implement this and deletion and text infilling
	return input_ids

    def add_rolling_noise(self, tokens):
	offset = np.random.randint(1, max(1, tokens.size(-1) - 1) + 1)
        tokens = torch.cat(
            (tokens[0:1], tokens[offset:-1], tokens[1:offset], tokens[-1:]),dim=0,)
        return tokens

    def transform_func(source, target):
        return source, target
