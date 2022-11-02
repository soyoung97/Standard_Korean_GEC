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
from scripts.extractPos.align import align
import re

path = "/Users/soyoung/Desktop/coding/research/GEC/GEC-Korean/extract_data/"
data_pool = ['wiki3', 'wiki4'] #['korean_learner', 'lang8', 'native'] # 'wiki1', 'wiki2', 'wiki3', 'wiki4']

def get_error_percentage_by_data(path, data_type, verbose=False):
    print(f"Processing {data_type} for get_error_percentage ... ")
    orig_ws, cor_ws, real_orig_ws, real_cor_ws, wrong = 0, 0, 0, 0, 0
    text = ''
    if 'wiki' in data_type:
        sub_dir = 'wiki'
    else:
        sub_dir = data_type
    with open(f"{path}/{sub_dir}/{data_type}.txt", 'r') as f:
        data = f.read().split("\n")
    data = [d.split("\t") for d in data if d != ""]
    data = [d for d in data if len(d) == 2]
    for idx, (orig, cor) in enumerate(data):
        # strip special tokens - applied only for lang8
        if data_type == 'lang8':
            orig = re.sub(' +', ' ', re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', orig))
            cor = re.sub(' +', ' ', re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', cor))
        orig = orig.strip()
        cor = cor.strip()
        orig_ws, cor_ws = 0, 0
        try:
            orig_tokens, cor_tokens = align(orig, cor)
        except:
            temp = f"{idx}\tError occured on align\t{orig}\t{cor}\n"
            text += temp
            print(temp)
            continue
        for orig_tok in orig_tokens:
            if orig_tok.pos == 'WS':
                orig_ws += 1
        for cor_tok in cor_tokens:
            if cor_tok.pos == 'WS':
                cor_ws += 1
        real_orig_ws = len(orig.split(' ')) - 1
        real_cor_ws = len(cor.split(' ')) - 1
        if orig_ws != real_orig_ws:
            temp = f"{idx}\t{real_orig_ws}!={orig_ws}\t{orig}\t{orig_tokens}\n"
            if verbose:
                print(temp)
            text += temp
            wrong += 1
        if cor_ws != real_cor_ws:
            temp = f"{idx}\t{real_cor_ws} != {cor_ws}\t{cor}\t{cor_tokens}\n"
            if verbose:
                print(temp)
            text += temp
            wrong += 1
    total = len(data) * 2
    out = f"{data_type}\nTotal: {total}, wrong: {wrong}\n{(wrong*100)/total}%\n"
    text = out + text
    with open(f"ws_error/{data_type}.out", 'w') as f:
        f.write(text)
    print(out)


for d in data_pool:
    get_error_percentage_by_data(path, d, verbose=True)
