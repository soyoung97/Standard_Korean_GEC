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
# coding=utf-8

from konlpy.tag import Kkma
from pprint import pprint
import numpy as np
from soynlp.hangle import jamo_levenshtein
from scripts.extractPos.tokens import Token, AlignedToken, POS_WHITESPACE
from scripts.extractPos.utils import break_down_to_jamo

ADD_LEFT = 1
ADD_RIGHT = 2
MATCH = 3
kkma = Kkma()

def min_edit_dist(a, b):
  d = np.zeros([len(a) + 1, len(b) + 1], dtype=int)
  v = np.zeros([len(a) + 1, len(b) + 1], dtype=int)
  for i in range(d.shape[0]):
    d[i][0] = i
  for j in range(d.shape[1]):
    d[0][j] = j
  for i in range(1, d.shape[0]):
    for j in range(1, d.shape[1]):
      d[i][j], v[i][j] = d[i - 1][j] + 1, ADD_LEFT
      if d[i][j] > d[i][j - 1] + 1:
        d[i][j], v[i][j] = d[i][j - 1] + 1, ADD_RIGHT
      if a[i - 1] == b[j - 1] and d[i][j] > d[i - 1][j - 1]:
        d[i][j], v[i][j] = d[i - 1][j - 1], MATCH
  return d, v


def align_pos_simple(original, corrected):
    orig_out = [kkma.pos(token) for token in original.strip().split(' ')]
    cor_out = [kkma.pos(token) for token in corrected.strip().split(' ')]
    return orig_out, cor_out


def align_pos_with_exceptions(original, corrected):
    orig_out, cor_out = align_pos(original, corrected)
    i = 0
    #origtok = original.split(' ')
    #cortok = corrected.split(' ')
    #while i < len(origtok):
    #    if origtok[i] == '':
    #        orig_out = orig_out[:i] +[[('', 'SW')]] + orig_out[i:]
    #    i += 1
    #i = 0
    #while i < len(cortok):
    #    if cortok[i] == '':
    #        cor_out = cor_out[:i] + [[('', 'SW')]] + cor_out[i:]
    #    i += 1
    space_orig, space_cor = len(original.split(' ')), len(corrected.split(' '))
    if len(orig_out) != space_orig:
        #print(original)
        #pprint(orig_out)
        orig_out = align_exceptions(orig_out, original.split(" "))
        #pprint(orig_out)
        #print()
    if len(cor_out) != space_cor:
        #print(corrected)
        #pprint(cor_out)
        cor_out = align_exceptions(cor_out, corrected.split(" "))
        #pprint(cor_out)
        #print()
    return orig_out, cor_out

def find_first_idx(lev, value):
    for i, v in enumerate(lev):
        if v >= value:
            return i
    return len(lev)

def align_exceptions(align_out, tokens): # "게" => "것이", '케': "하게" : 항상 그럼.
    diff = len(tokens) - len(align_out)
    assert diff > 0  # assumption that wrong alignment is always shorter
    for _ in range(diff):
        lev = []
        for i, info in enumerate(align_out):
            composed_str = "".join([x[0] for x in info])
            orig_str = tokens[i]
            lev.append(jamo_levenshtein(composed_str, orig_str))
        # iterate through levenshtein distance to find the tobe_splitted info.
        # Tobe_splitted is the first one that has large lev distance (e.g. > 2.5)
        mixed_idx = find_first_idx(lev, 2)
        if mixed_idx == len(lev): # Nothing was found
            mixed_idx = find_first_idx(lev, max(lev))
            if mixed_idx == len(lev):
                mixed_idx -= 1
        align_cor = align_out[:mixed_idx] # get left previous alignment info
        tobe_splitted = align_out[mixed_idx]
        to_match = tokens[mixed_idx + 1]
        match_scores = []
        for i in range(len(tobe_splitted)):
            match_scores.append(jamo_levenshtein("".join([x[0] for x in tobe_splitted[i:]]), to_match))
        split_point = match_scores.index(min(match_scores))
        # Exceptional cases:
        if split_point != 0 and tobe_splitted[split_point-1][0] == '하':
            split_point -= 1
        # [] 들어가는 것 방지
        if len(tobe_splitted) == 1:
            tobe_splitted = [tuple([tobe_splitted[0][0][0].strip(), tobe_splitted[0][1]]), tuple([tobe_splitted[0][0][1:].strip(), tobe_splitted[0][1]])]
            split_point = 1
        elif split_point == 0:
            split_point = 1
        elif split_point == len(tobe_splitted):
            split_point -= 1
        align_cor.append(tobe_splitted[:split_point])
        align_cor.append(tobe_splitted[split_point:])
        align_cor += align_out[mixed_idx+1:]
        align_out = align_cor
    return align_out

def parse(sentence):
  tokens = [Token(token[0], token[1]) for token in kkma.pos(sentence)]

  jamo_level_tokens = ''.join([token.jamo for token in tokens])
  jamo_level_sentence = break_down_to_jamo(sentence)
  _, v = min_edit_dist(jamo_level_tokens, jamo_level_sentence)
  tokens_with_whitespace = []
  token_index = len(tokens) - 1
  token_size = len(tokens[token_index].jamo)
  i, j = v.shape[0] - 1, v.shape[1] - 1
  while i:
    if v[i][j] == ADD_LEFT:
      i, token_size = i - 1, token_size - 1
    elif v[i][j] == ADD_RIGHT:
      j = j - 1
    else:
      i, j, token_size = i - 1, j - 1, token_size - 1
    if not token_size:
      tokens_with_whitespace.append(tokens[token_index])
      if j > 1 and jamo_level_sentence[j - 1] == ' ':
        tokens_with_whitespace.append(Token(' ', POS_WHITESPACE))
        j -= 1
      token_index -= 1
      token_size = len(tokens[token_index].jamo)
  tokens_with_whitespace.reverse()

  return tokens_with_whitespace

def align_pos(original_sentence, corrected_sentence): # should be used when align function works properly.
    orig, cor = align(original_sentence, corrected_sentence)
    orig_out, cor_out = [[]], [[]]
    for tok in orig:
        if tok.pos == 'WS':
            orig_out.append([])
        else:
            orig_out[-1].append((tok.token, tok.pos))
    for tok in cor:
        if tok.pos == 'WS':
            cor_out.append([])
        else:
            cor_out[-1].append((tok.token, tok.pos))
    #print(f"Algin_pos: \n{original_sentence}\n{orig_out}\n\n{corrected_sentence}\n{cor_out}")
    return orig_out, cor_out

def align(original_sentence, corrected_sentence):
  original_tokens = parse(original_sentence)
  corrected_tokens = parse(corrected_sentence)
  return (original_tokens, corrected_tokens)

if __name__ == '__main__':
  #align_pos_with_exceptions(u'보통 무서우니까 한국 사람들에게 한국어로 말을 못 하지만 술을 마실 때 한국 친구들에게 한국어로 많이 많이 말할 수 있어서 금요일마다 한국어를 공부하고 있어요 ~ ^ ^ ㅋㅋ 토요일 밤마다 영국 친구들 만나서 신촌에서 나무 카페에 가고 같이 공부하고 숙제를 해요 .', '이 세상에 살면서 갈 수 있는 길도 많거니와 할 수 있는 일도 여러 가지지만 아쉽게도 이 사회가 우리에게 미친 영향 때문에 다른 길로 벗어나는 게 생각보다 힘듭니다 .')
  original_sentence = u'공부를 하면할수록 모르는게 많다는 것을 알게 됩니다.'
  corrected_sentence = u'공부를 하면 할 수록 모르는게 많다는 것을 알게 됩니다.'

  original_tokens, corrected_tokens = align(original_sentence, corrected_sentence)
  print('Original sentence:', original_sentence)
  pprint(original_tokens)
  print('Corrected sentence:', corrected_sentence)
  pprint(corrected_tokens)
