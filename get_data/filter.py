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
'''
Description: This file preprocesses and make Lang-8 dataset.

Usage: python filter.py -d [file path] -o [file path]
-d: pickled result file path after running extract_err-cor-pair.py
-o: file path to save the result of this script
'''
from tqdm import tqdm
import pandas as pd
import re
import argparse
import string
import math
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
from collections import Counter
from hangul_util import *
from soynlp.hangle import jamo_levenshtein



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datafile', type=str, default='lang8_raw.txt')
    parser.add_argument('-o', '--output', type=str, default='lang8_filtered.txt')
    parser.add_argument('-c', '--cleaned_output', type=str, default='lang8.txt')
    args = parser.parse_args()
    return args

def load_data(file_path):
    original = []
    corrected = []
    with open(file_path, 'r') as f:
        data = [x for x in f.read().split('\n') if x != '']
        for line in tqdm(data):
            try:
                o, c = line.split('\t')
            except:
                continue
            original.append(o)
            corrected.append(c)
    return original, corrected

def get_df(data):
    original = [x[0] for x in data]
    corrected = [x[1] for x in data]
    df = pd.DataFrame({'original': original, 'corrected': corrected})
    return df

def drop_other_lang(data):
    hangul = re.compile('[^ ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z@#$%&\(\){|}\[\]\/\-><=_:;.,?!~\^"（）\'‘’]+')
    only_hangul = re.compile('[ㄱ-ㅎㅏ-ㅣ가-힣]+')
    # If there is data that contains other characters other than hangul, the findall returns a list of that.
    # Example: डीटीसी की जो बसें सु, or ờ á Đấ à ộ ự ệ ố ế
    data = [x for x in data if len(hangul.findall(x[0])) == 0 and len(hangul.findall(x[1])) == 0]
    data = [x for x in data if len(only_hangul.findall(x[0])) != 0 and len(only_hangul.findall(x[1])) != 0]
    return data

def token_too_long(org, cor):
    orgs = np.array([len(x) for x in org.split(" ")])
    cor = np.array([len(x) for x in cor.split(" ")])
    if max(max(orgs), max(cor)) > 20:
        return True
    return False


def calc_vars(data):
    tok_path = get_tokenizer()
    sp = SentencepieceTokenizer(tok_path)
    txt = [{
        'len_token_o': len(sp(x[0])), 'len_token_c': len(sp(x[1])), 'len_char_o': len(x[0]), 'len_char_c': len(x[1]),
        #'jamo_levenshtein': jamo_levenshtein(x[0], x[1]), 
        'o': x[0], 'c': x[1], 'token_div': None, 'char_div': None, 'token_min': None
        } for x in data]
    for info in txt:
        info['token_div'] = info['len_token_c']/info['len_token_o']
        info['char_div'] = info['len_char_c']/info['len_char_o']
        info['token_min'] = min(info['len_token_c'], info['len_token_o'])
        #info['token_log'] = math.log(info['token_min'], 20)
        #info['ratio'] = info['jamo_levenshtein'] / info['token_min'] * info['token_log']
    # apply cond
    try:
        txt = np.array(txt, dtype=object)
    except:
        import pdb; pdb.set_trace()
    const = np.array([(0.25 < x['token_div'] < 4) & (0.5 < x['char_div'] < 1.25) & (x['token_min'] > 5)
        & ('or' not in x['c']) & ('good' not in x['c']) & (not token_too_long(x['o'], x['c'])) for x in txt])
    data = [[x['o'].strip(), x['c'].strip()] for x in txt[const]]
    return data

def cleanup_corpus(data):
    cleaned_src = [re.sub(r'([{}])'.format(string.punctuation),r' \1 ', x[0]) for x in data]
    cleaned_src = [re.sub(' +', ' ', x) for x in cleaned_src]
    cleaned_tgt = [re.sub(r'([{}])'.format(string.punctuation), r' \1 ', x[1]) for x in data]
    cleaned_tgt = [re.sub(' +', ' ', x) for x in cleaned_tgt]
    cleaned = [(s, t) for s, t in zip(cleaned_src, cleaned_tgt) if '/' not in s and '/' not in t]
    # remove samples with \
    lang8_count = Counter(cleaned)
    first_dups = [pair for pair,cnt  in lang8_count.items() if cnt > 1]
    print(len(first_dups))
    print('lang8 counter:', len(lang8_count.keys()))
    uniq = list(set(cleaned))
    print('unique lang8 data:', len(uniq))
    return uniq

def write(output, cleaned_unique_corpus):
    cnt = 0
    with open(output, 'w', encoding='utf8') as f:
        f.write('src' + '\t' + 'tgt' + '\n')
        for pair in cleaned_unique_corpus:
            if pair[0] != pair[1]:
                f.write('\t'.join(pair).strip() + '\n')
                cnt += 1

    print('Saved dataset:', cnt)


# flatten datasets that have only one lines per each data sample.
def split_sentences(orig, cor):
    res = []
    for o, c in zip(orig, cor):
        if o.count('.') > 1:
            orig_sents, cor_sents = o.split('.'), c.split('.')
            if len(orig_sents) == len(cor_sents):
                for o_sub, c_sub in zip(orig_sents[:-1], cor_sents[:-1]):
                    if o_sub != c_sub:
                        if min(len(o_sub), len(c_sub)) > 4:
                            res.append([o_sub + ' .', c_sub + ' .'])
        else:
            res.append([o, c])
    return res

def remove_misfunctioned_cases(data):
    data = ([d for d in data if len(d) == 2])
    data = [d for d in data if len(d[0]) > 2]
    return [d for d in data if len(d[1]) > 2]


def only_leave_unique(data):
    data = [x for x in data if x[0] != x[1]]
    return data

def cleanup_punct_and_brackets(cleaned_list):
    orig = [x[0].replace('^', '') for x in cleaned_list]
    cor = [x[1].replace('^', '') for x in cleaned_list]
    # remove (in order)
    # 1 ) 2 ) ....
    # (Inside paranthesis)
    # <Inside brackets>
    # Duplicates of ㄱ > ; ㅠ ㅜ ? ! . ~
    orig = [re.sub(r'( )(?=\1)', '', re.sub(r'\[.*?\]', '', re.sub(r'\<.*?\>', '', re.sub(r'(\d+ )\)', '', re.sub(r'\([^)]*\)', '', re.sub(r'(~)(?=\1)', '', re.sub(r'(ㄱ)(?=\1)', '', re.sub(r'( >)(?=\1)', '', re.sub(r'( ;)(?=\1)', '', re.sub(r'(ㅠ)(?=\1)', '', re.sub(r'(ㅜ)(?=\1)', '', re.sub(r'( ?)(?=\1)', '', re.sub(r'( !)(?=\1)', '', re.sub(r'( .)(?=\1)', '', x)))))))))))))).strip() for x in orig]
    cor = [re.sub(r'( )(?=\1)', '', re.sub(r'\[.*?\]', '', re.sub(r'\<.*?\>', '', re.sub(r'(\d+ )\)', '', re.sub(r'\([^)]*\)', '', re.sub(r'(~)(?=\1)', '', re.sub(r'(ㄱ)(?=\1)', '', re.sub(r'( >)(?=\1)', '', re.sub(r'( ;)(?=\1)', '', re.sub(r'(ㅠ)(?=\1)', '', re.sub(r'(ㅜ)(?=\1)', '', re.sub(r'( ?)(?=\1)', '', re.sub(r'( !)(?=\1)', '', re.sub(r'( .)(?=\1)', '', x)))))))))))))).strip() for x in cor]
    return orig, cor

def final_cleanup_and_write(filename, res):
    orig = res
    res = [x for x in res if ('\t' not in x[0] and '\t' not in x[1] and len(x[0]) >= 2 and len(x[1]) >= 2)]
    print("Length before levenshtein: ", len(res))
    res = [x for x in res if jamo_levenshtein(x[0], x[1]) > 10]
    print("Length after levenshtein: ", len(res))
    data = '\n'.join(['\t'.join(x) for x in res])
    with open('kor_lang8.txt', 'w') as f:
        f.write(data)




def main():
    args = parse_args()

    print('1. Load data')
    org, cor = load_data(args.datafile)
    print('dataset length', len(org))

    print('2. Remove white spaces, misfunctioned cases, and other languages')
    # remove other languages
    org = [re.sub('[a-zA-Z]+', '', re.sub(u'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uff9f\u4e00-\u9faf\u3400-\u4dbf]', '', x)) for x in org]
    cor = [re.sub('[a-zA-Z]+', '', re.sub(u'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uff9f\u4e00-\u9faf\u3400-\u4dbf]', '', x)) for x in cor]

    org, cor = [x.strip() for x in org], [x.strip() for x in cor]
    res = remove_misfunctioned_cases(zip(org, cor))
    print("dataset length", len(res))

    print('3. Drop rows of other languages')
    res = drop_other_lang(res)

    print('4. Calculate variables and apply conditions')
    res = calc_vars(res)
    print("dataset length", len(res))

    print('5. Save cleaned up data')
    cleaned_list = cleanup_corpus(res)
    # changed to list
    orig, cor = cleanup_punct_and_brackets(cleaned_list)
    res = split_sentences(orig, cor)
    res = only_leave_unique(res)
    res = final_cleanup_and_write('lang8.txt', res)
    print('Done!')

if __name__ == '__main__':
    main()

