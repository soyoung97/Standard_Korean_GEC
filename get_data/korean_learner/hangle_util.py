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
import numpy as np

kor_begin, kor_end = (44032, 55203)
jaum_begin, jaum_end = (12593, 12622)
moum_begin, moum_end = (12623, 12643)
chosung_base  = 588
jungsung_base = 28

chosung_list = [ 'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ',
        'ㅅ', 'ㅆ', 'ㅇ' , 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ',
        'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ',
        'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ',
        'ㅡ', 'ㅢ', 'ㅣ']

jongsung_list = [
    ' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ',
        'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ',
        'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ',
        'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jaum_list = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ',
              'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ',
              'ㅃ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

moum_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
              'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']


def compose(chosung, jungsung, jongsung):
    return chr(kor_begin + chosung_base * chosung_list.index(chosung) + jungsung_base * jungsung_list.index(jungsung) + jongsung_list.index(jongsung))

def decompose(c):
    # print("input char >>> ", c)
    if len(c) > 1 :
        return [ decompose(d) for d in c ]

    if not character_is_korean(c):
        return ('','','')
    i = ord(c)
    if (jaum_begin <= i <= jaum_end):
        return (c, ' ', ' ')
    if (moum_begin <= i <= moum_end):
        return (' ', c, ' ')
    i -= kor_begin
    cho  = i // chosung_base
    jung = ( i - cho * chosung_base ) // jungsung_base
    jong = ( i - cho * chosung_base - jung * jungsung_base )
    return (chosung_list[cho], jungsung_list[jung], jongsung_list[jong])

def character_is_korean(c):
    i = ord(c)
    return (kor_begin <= i <= kor_end) or (jaum_begin <= i <= jaum_end) or (moum_begin <= i <= moum_end)

def is_jaum(c):
    return c in jaum_list

def merge_jaum_token(i, tok):
    print('jaum in ',i, 'th in tok : ', tok )
    cho,jung,jong = decompose(tok[i-1])
    rep = compose(cho,jung,tok[i])
    new_tok = tok[:i-1] + rep + tok[i+1:]
    return new_tok


def check_jaum_token(tok):
    if not tok :
        return tok
    for i, t in enumerate(tok):
        if is_jaum(t) and i != 0 and len(tok) > 1:
            return check_jaum_token(merge_jaum_token(i, tok))

    return tok

def merge_two_token(left,right):
    print('left:',left,'\tright:',right)
    if is_jaum(right[0]):
        cho,jung,_ = decompose(left[-1])
        print(f'compse target >>> {left} | {cho},{jung},{right[0]}')
        new_left_last = compose(cho,jung,right[0])
        newtok = left[:-1]+new_left_last+right[1:]
        return newtok, None
    else :
        return left, right

def apply_short_rule(tok):
    if not tok:
        return tok
    start = tok
    while True:
        tmp = rule_34(start)
        tmp = rule_35(tmp)
        tmp = rule_36(tmp)
        tmp = rule_thumb(tmp)
        if start != tmp :
            start = tmp
        else :
            break
    return start

def rule_34(tok):

    decomp = decompose(tok)
    if not isinstance(decomp, list):
        decomp = [decomp]
    print("check:RULE34, decomposed >>> ",decomp)
    for i in range(len(decomp)-1, 0, -1):
        (a, b, c) = decomp[i]
        if a == 'ㅇ' and b in ['ㅏ','ㅓ','ㅐ','ㅔ', 'ㅕ'] and c == 'ㅆ' :
            p_a,p_b,p_c = decomp[i-1]
            if p_b in ['ㅏ','ㅓ','ㅕ','ㅐ','ㅔ'] and p_c == ' ':
                if p_a == 'ㅎ' and b in ['ㅕ', 'ㅏ']:
                    new_tok = tok[:i-1]+ compose(p_a, 'ㅐ', c) + tok[i+1:]
                else:
                    new_tok = tok[:i-1]+ compose(p_a, p_b, c) + tok[i+1:]
                return new_tok
        elif a == 'ㅇ' and b in ['ㅏ','ㅓ','ㅐ','ㅔ', 'ㅕ'] :
            p_a,p_b,p_c = decomp[i-1]
            if p_b in ['ㅏ','ㅓ','ㅕ','ㅐ','ㅔ'] and p_c == ' ':
                if p_a == 'ㅎ' and b in ['ㅕ', 'ㅏ'] :
                    new_tok = tok[:i-1]+ compose(p_a, 'ㅐ', c) + tok[i+1:]
                else:
                    new_tok = tok[:i-1]+ compose(p_a, p_b, c) + tok[i+1:]
                return new_tok

    return tok

def rule_35(tok):
    decomp = decompose(tok)
    if not isinstance(decomp, list):
        decomp = [decomp]
    print("check:RULE35, decomposed >>> ", decomp)

    for i in range(len(decomp) - 1, 0, -1):
        (a, b, c) = decomp[i]
        if a == 'ㅇ' and b in ['ㅏ', 'ㅓ'] and c == 'ㅆ':
            p_a, p_b, p_c = decomp[i - 1]
            if p_b in ['ㅗ', 'ㅜ', 'ㅚ'] and p_c == ' ':
                comp = compose_moum(p_b,b)
                print("composed moum {}+{} ->{}".format(p_b,b,comp))
                if comp :
                    new_tok = tok[:i - 1] + compose(p_a, comp, c) + tok[i + 1:]
                    return new_tok
            elif p_a == 'ㄴ' and p_b == 'ㅗ' and p_c == 'ㅎ':
                new_tok = tok[:i-1] + compose(p_a, 'ㅘ', c) + tok[i + 1:]
                return new_tok

        # elif a == 'ㅇ' and b in ['ㅏ', 'ㅓ']:
        #     p_a, p_b, p_c = decomp[i - 1]
        #     if p_b in ['ㅗ', 'ㅜ', 'ㅚ'] and p_c == ' ':
        #         comp = compose_moum(p_b, b)
        #         print("composed moum {}+{} ->{}".format(p_b, b, comp))
        #         if comp :
        #             new_tok = tok[:i - 1] + compose(p_a, comp, c) + tok[i + 1:]
        #             return new_tok
        #     elif p_a == 'ㄴ' and p_b == 'ㅗ' and p_c == 'ㅎ':
        #         new_tok = tok[:i-1] + compose(p_a, 'ㅘ', c) + tok[i + 1:]
        #         return new_tok

    return tok

def rule_36(tok):
    decomp = decompose(tok)
    if not isinstance(decomp, list):
        decomp = [decomp]
    print("check:RULE36, decomposed >>> ", decomp)

    for i in range(len(decomp) - 1, 0, -1):
        (a, b, c) = decomp[i]
        if a == 'ㅇ' and b in ['ㅓ'] and c == 'ㅆ':
            p_a, p_b, p_c = decomp[i - 1]
            if p_b in ['ㅣ'] and p_c == ' ':
                comp = compose_moum(p_b, b)
                print("composed moum {}+{} ->{}".format(p_b, b, comp))
                if comp:
                    new_tok = tok[:i - 1] + compose(p_a, comp, c) + tok[i + 1:]
                    return new_tok

    return tok

def rule_thumb(tok):

    print("check:RULEㅗ, decomposed >>> ", tok)

    for i in range(0, len(tok)):
        if tok[i] in ['렵','럽'] and i+1 < len(tok) and tok[i+1]=='어':
            middle= '려' if tok[i]=='렵' else '러'
            return tok[:i] + middle + '워'
    return tok

moum_set={
    ('ㅗ','ㅏ') : "ㅘ",
    ('ㅜ','ㅓ') : "ㅝ",
    ('ㅚ','ㅓ') : "ㅙ",
    ('ㅣ','ㅓ') : 'ㅕ'
}

def compose_moum(a,b):
    return moum_set.get((a,b), None)


def levenshtein(s1, s2, cost=None):
    # based on Wikipedia/Levenshtein_distance#Python
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    if not cost:
        cost = {}

    def get_cost(c1, c2, cost):
        return 0 if (c1 == c2) else cost.get((c1, c2), 1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + get_cost(c1, c2, cost)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def jamo_levenshtein(s1, s2):
    if len(s1) < len(s2):
        return jamo_levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    def get_jamo_cost(c1, c2):
        c1_is_korean, c2_is_korean = character_is_korean(c1), character_is_korean(c2)
        if c1_is_korean and c2_is_korean:
            return 0 if (c1 == c2) else levenshtein(decompose(c1), decompose(c2))
        elif not (c1_is_korean) and not (c2_is_korean):
            return 0 if (c1 == c2) else 1
        else:
            return 1

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + get_jamo_cost(c1, c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

if __name__ =='__main__' :
    test_set = [
        '것이ㄴ지',#0
        '전하지ㄴ',#1
        '편이ㄴ',#2
        '살ㄴ',#3
        '나ㄴ다',#4
        '피우ㄴ다',#5
        '만나ㄹ',#6
        '행복하아하ㄹ',#7
        '만들ㄹ',#8
        '처리하ㄹ',#9
        '다루어지ㄹ',#10
        '하ㄴ',#11
        '마',#12
        'ㄴ다'#13

    ]
    # for t in test_set:
    #     print(t, check_jaum_token(t))

    # for i,(a,b) in enumerate(zip(test_set, test_set[1:])):
    #     c,d = merge_two_token(a,b)
    #     print(i,a,b,c,d)

    set_34 = ['가았다', '나았다', '타았다','서었다','켜었다','펴었다'
                 ,'가아','나아','서어','켜어','펴어'
                 ,'개었다','내었다','베었다','세었다'
                 ,'개어','내어','베어','세어'
                 ,'하였다','더하였다','흔하였다'
                 ,'하여','더하여','흔하여'
                 ,'따았다','따아'
                 ,'건너어도','건너었다'
                 ,'일어났어요'
                 ]

    set_35 = [
        '놓아', '놓아라', '놓았다',
        '꼬아','보아','쏘아','두어','쑤어','주어','추어','추어서','추어야',
        '꼬았다','보았다','쏘았다','두었다','두었다','주었다','추었다',
        '괴어','되어','뵈어','쇠어','쐬어',
        '괴었다','되었다','뵈었다','쇠었다','쐬었다'

    ]

    set_36 = [
        '가지었다','견디었다','다니었다','막히었다','버티었다','치이었다',
        '숙이었다','잡히었다',''
    ]

    set_thumb = [
        '가게이었습니다',
        '관광사업이'
    ]

    for t in set_thumb:
        print(t,apply_short_rule(t))
