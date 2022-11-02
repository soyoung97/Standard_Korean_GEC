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
import re
from sklearn.model_selection import train_test_split
import random
import os

def main():
    while True:
        actions = input("Select actions from the following: (punct_split, merge_wiki, split_pairs, train_split, q, merge_files, reorder_m2, make_sample, make_union) : ")
        if actions == "merge_wiki":
            merge_wiki()
        elif actions == 'punct_split':
            punct_split()
        elif actions == "make_union":
            make_union()
        elif actions == "split_pairs":
            pair_path = input("Please specify the pair file path: ")
            split_pairs(pair_path)
        elif actions == 'train_split':
            whole_path = input("Please specify the whole pair data path: ")
            train_test_val_split(whole_path)
        elif actions == 'merge_files':
            paths = []
            while True:
                text = input("Please add file path, q when done: ")
                if text == 'q':
                    break
                paths.append(text)
            merge_files(paths)
        elif actions == 'reorder_m2':
            m2_file_path = input("Please write the full path for full m2 file.: ")
            align_file_path = input("Please write the common path for train/test/val file to be aligned.: ")
            reorder_m2(m2_file_path, align_file_path)
        elif actions == 'make_sample':
            types = input("Which sample? (in number) 1: korean_learner, 2:lang8, 3:native, 4:wiki ")
            types = ['', "korean_learner", "lang8", "native", "wiki"][int(types)]
            make_sample(types)
        elif actions == 'q':
            print("Bye!")
            break
        else:
            print("Typed word does not match any of available instructions: please write again.")

def punct_split():
    data_path = input("Data path: ")
    pat = re.compile(r"([-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》])")
    with open(data_path, 'r') as f:
        data = f.read().split("\n")
    if data[-1] == '':
        data = data[:-1]
    punct_splitted = [re.sub(' +', ' ', pat.sub(" \\1 ", x)).strip() for x in data]
    data = punct_splitted
    with open(data_path.split(".txt")[0] + "_punct.txt", 'w') as f:
        f.write("\n".join(data))
    print("Punctuation split done!")

def make_union():
    if os.path.exists("../korean_learner/korean_learner_train.txt"):
        cd = '../'
    elif os.path.exists("extract_data/korean_learner/korean_learner_train.txt"):
        cd = 'extract_data/'
    for mode in ['train', 'test', 'val']:
        os.system(f"echo > {cd}union/union_{mode}.txt") # file cleanup
        os.system(f"echo > {cd}union/union_{mode}.m2") # file cleanup

        for data in ['lang8', 'korean_learner', 'native']:
            syscommand = f"cat {cd}{data}/{data}_{mode}.txt >> {cd}union/union_{mode}.txt"
            os.system(syscommand)
            os.system(f"echo >> {cd}union/union_{mode}.txt")
            os.system(f"cat {cd}{data}/{data}_{mode}.m2 >> {cd}union/union_{mode}.m2")
    for mode in ['train', 'test', 'val']:
        os.system(f"cat {cd}/union/union_train.txt {cd}/union/union_test.txt {cd}/union/union_val.txt > {cd}/union/union.txt")
        os.system(f"cat {cd}/union/union_train.m2 {cd}/union/union_test.m2 {cd}/union/union_val.m2 > {cd}/union/union.m2")
    split_pairs(f"{cd}/union/union.txt")
    split_pairs(f"{cd}/union/union_val.txt")
    print("make union done")


def reorder_m2(m2_file_path, align_file_path):
    with open(m2_file_path, 'r') as f:
        data = f.read().split("\nS ")
    key = [x.split('\nA')[0] for x in data]
    key[0] = key[0][2:]
    data[0] = data[0][2:]
    if data[-1][-2] == '\n':
        data[-1] = data[-1][:-1]
    m2_dict = {x.strip(): y for x, y in zip(key, data)}
    for path in ['val', 'test', 'train']:
        res = []
        with open(f"{align_file_path}_{path}.txt", "r") as f:
            align_file = [x.split("\t")[0].strip() for x in f.read().split('\n')]
        for keytxt in align_file:
            out = m2_dict.get(keytxt)
            if out is None:
                print("Cannot get output!")
                import pdb; pdb.set_trace()
            else:
                res.append(out)
        with open(f"{align_file_path}_{path}.m2", "w") as f:
            txt = 'S ' + '\nS '.join(res)
            f.write(txt)
        print(f"{path} m2 file written done.")


def write(data_list, file_path):
    with open(file_path, 'w') as f:
        f.write('\n'.join(data_list))

def train_test_val_split(whole_path):
    start_path = whole_path.split('.txt')[0]
    with open(whole_path, 'r') as f:
        data = [d for d in f.read().split('\n') if d != '']
    train, val_test = train_test_split(data, test_size=0.3, random_state = 1)
    val, test = train_test_split(val_test, test_size=0.5, random_state = 1)
    write(train, start_path + '_train.txt')
    write(val, start_path + '_val.txt')
    write(test, start_path + '_test.txt')
    print(f"Length = train: {len(train)}, val: {len(val)}, test: {len(test)}, write complete")


def split_pairs(pair_path):
    start_path = pair_path.split('.txt')[0]
    with open(pair_path, 'r') as f:
        data = [x.split('\t') for x in f.read().split("\n") if x != '']
    orig = [x[0].strip() for x in data]
    corrected = [x[1].strip() for x in data]
    with open(f"{start_path}_original.txt", 'w') as f:
        f.write("\n".join(orig))
    with open(f"{start_path}_corrected.txt", 'w') as f:
        f.write("\n".join(corrected))
    print(f"{start_path}_original.txt and {start_path}_corrected.txt write complete")
    
def merge_files(paths):
    data = []
    for path in paths:
        with open(path, 'r') as f:
            data.append(f.read())
    data = ''.join(data)
    out_path = input("Please specify output file path: ")
    with open(out_path, 'w') as f:
        f.write(data)
    print("Merge files done!")

def merge_wiki():
    data = []
    for filename in ['wiki1', 'wiki2', 'wiki3', 'wiki4']:
        with open(f'../wiki/{filename}.txt', 'r') as f:
            data.append(f.read())
    data = ''.join(data)
    with open("../wiki/wiki.txt", 'w') as f:
        f.write(data)
    print("Merge wiki done!")

def prepare_wiki_cloud():
    data = []
    for filename in ['wiki1', 'wiki2', 'wiki3', 'wiki4']:
        with open(f'extract_data/wiki/{filename}.txt', 'r') as f:
            data.append(f.read())
    data = ''.join(data)
    with open("extract_data/wiki/wiki.txt", 'w') as f:
        f.write(data)
    print("Merge wiki done!")
    train_test_val_split("extract_data/wiki/wiki.txt")

def make_sample(types):
    random.seed(0)
    with open(f"../{types}/{types}.txt", "r") as f:
        data = [x for x in f.read().strip().split("\n") if x != '']
    random.shuffle(data)
    with open(f"../../sample_data/{types}.txt", "w") as f:
        f.write("\n".join(data[:500]).strip())
    split_pairs(f"../../sample_data/{types}.txt")


if __name__ == '__main__':
    #reorder_m2("../../src/errant/edit-extraction/alignment_output/korean_learner_full.m2", '../korean_learner/korean_learner')
    #reorder_m2("../../src/errant/edit-extraction/alignment_output/lang8_full.m2", '../lang8/lang8')
    #reorder_m2("../../src/errant/edit-extraction/alignment_output/native_full.m2", '../native/native')
    #prepare_wiki_cloud()
    main()
