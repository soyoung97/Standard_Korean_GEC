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
from itertools import groupby
import scripts.rdlextra as DL
from pos_granularity import adjust_pos_granularity, composite_error_type, PARTICLE, NOUN, VERB, ADJECTIVE, UNCLASSIFIED, CONJUGATION
import string
from soylemma import Lemmatizer
from pprint import pprint

# Some global variables
# TODO: adjust
KOREAN_CONTENT_POS = ['NNG', 'NNP', 'NNB', 'NNM', 'NR', 'NP', # 체언
'VV', 'VA', 'VXV', 'VXA', 'VCP', 'VCN', # 용언
'MDT', 'MDN', 'MAG', 'MAC'] # 관형사, 부사
PUNCT = set([x for x in """-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》"""])
SPECIAL_CODE = ['WS', 'T2', 'T3']
SPC2CODE = {
        'WS': 'WS',
        'T2': "WO",
        'T3': "WO"
}
lemmatizer = Lemmatizer()

### FUNCTIONS ###

def get_opcodes(alignment):
    s_start = 0
    s_end   = 0
    t_start = 0
    t_end   = 0
    opcodes = []
    for op in alignment:
        if op[0] == "D": # Deletion
            s_end += 1
        elif op[0] == "I": # Insertion
            t_end += 1
        elif op[0].startswith("T"): # Transposition
            # Extract number of elements involved (default is 2)
            k = int(op[1:] or 2)
            s_end += k
            t_end += k
        else: # Match or substitution
            s_end += 1
            t_end += 1
        # Save
        opcodes.append((op, s_start, s_end, t_start, t_end))
        # Start from here
        s_start = s_end
        t_start = t_end
    return opcodes

def merge_edits(edits, code='X'):
    if edits:
        return [(code, edits[0][1], edits[-1][2], edits[0][3], edits[-1][4])]
    else:
        return edits

def check_split(source, target, edits):
    gvs = []
    t = []
    # Collect the tokens
    for e in edits:
        try:
            s_tok = source[e[1]:e[2]].orth_.replace("'", "")
            t_tok = target[e[3]:e[4]].orth_.replace("'", "")
            if len(s_tok) >= 1: s.append(s_tok)
            if len(t_tok) >= 1: t.append(t_tok)
        except:
            import pdb; pdb.set_trace()
    if len(s) == len(t):
        return False
    elif len(s) == 1 and len(t) > 1:
        string = s[0]
        tokens = t
    elif len(t) == 1 and len(s) > 1:
        string = t[0]
        tokens = s
    else:
        return False
    # Check split
    if string.startswith(tokens[0]): # Matches beginning
        string = string[len(tokens[0]):]
        if string.endswith(tokens[-1]): # Matches end
            string = string[:-len(tokens[-1])]
            # Matches all tokens in the middle (in order)
            match = True
            for t in tokens[1:-1]:
                try:
                    i = string.index(t)
                    string = string[i+len():]
                except:
                    # Token not found
                    return False
            # All tokens found
            return True
    # Any other case is False
    return False

# allsplit: No edits are ever merged. Everything is 1:1, 1:0 or 0:1 only.
def get_edits_split(source, target, edits):
    new_edits = []
    for edit in edits:
        op = edit[0]
        if op != "M":
                new_edits.append(edit)
    return new_edits

"""
def merge_ws2(source, target, edits):
    # case where we need to ADD ws
    out_edits = []
    merged=False
    for idx, edit in enumerate(edits):
        if merged:
            merged=False
            continue
        if edit[0] == 'S':
            orig, cor = source[edit[1]:edit[2]], target[edit[3]:edit[4]]
            if len(orig) == 1 and len(cor) == 1 and orig[0] not in PUNCT and cor[0] not in PUNCT:
                orig, cor = orig[0], cor[0]
                if cor in orig and len(target) != edit[4]: # possibly 띄어쓰기를 "넣어준" 경우! Make sure that it's not the last one
                    if cor + target[edit[4]] == orig and (idx + 1) != len(edits):
                        out_edits.append(merge_edits(edits[idx:idx+2], code='WS')[0])
                        merged = True
                        continue
                elif orig in cor and len(source) != edit[2]:
                    if orig + source[edit[2]] == cor and (idx + 1) != len(edits):
                        out_edits.append(merge_edits(edits[idx:idx + 2], code='WS')[0])
                        merged=True
                        continue
        elif edit[0] == 'I' and idx != (len(edits) - 1) and edits[idx + 1][0] == 'S': # I than S인 경우의 direction도 생각해줘야함. Make sure it's not the last one.
            cor = target[edit[3]:edit[4]] # original is [].
            if len(cor) == 1 and cor[0] not in PUNCT:
                cor = cor[0]
                next_edit = edits[idx + 1]
                next_source, next_target = source[next_edit[1]:next_edit[2]], target[next_edit[3]:next_edit[4]]
                if next_source[0] == (cor + next_target[0]) and len(next_source) == 1 and len(next_target) == 1 and next_source[0] not in PUNCT and next_target[0] not in PUNCT:
                    out_edits.append(merge_edits(edits[idx:idx + 2], code='WS')[0])
                    merged = True
                    continue
        elif edit[0] == 'D' and idx != (len(edits) - 1) and edits[idx + 1][0] == 'S': # I than S인 경우의 direction도 생각해줘야함.
            orig = source[edit[1]:edit[2]] # corrected is [].
            if len(orig) == 1 and orig[0] not in PUNCT:
                orig = orig[0]
                next_edit = edits[idx + 1]
                next_source, next_target = source[next_edit[1]:next_edit[2]], target[next_edit[3]:next_edit[4]]
                if (orig + next_source[0]) == next_target[0] and len(next_source) == 1 and len(next_target) == 1 and next_source[0] not in PUNCT and next_target[0] not in PUNCT:
                    out_edits.append(merge_edits(edits[idx:idx + 2], code='WS')[0])
                    merged = True
                    continue
        out_edits.append(edit)
    return out_edits
"""

def merge_ws(source, target, edits):
    out_edits = []
    while len(edits) != 0:
        if len(edits) >= 3: # 3-way merge first
            pedit = merge_edits(edits[0:3], code='WS')[0]
            if ''.join(source[pedit[1]:pedit[2]]) == ''.join(target[pedit[3]:pedit[4]]):
                out_edits.append(pedit)
                edits = edits[3:]
                continue
        if len(edits) >= 2: # Do 2-way merge only if 3-way much was not possible.
            pedit = merge_edits(edits[0:2], code='WS')[0]
            if ''.join(source[pedit[1]:pedit[2]]) == ''.join(target[pedit[3]:pedit[4]]):
                out_edits.append(pedit)
                edits = edits[2:]
                continue
        out_edits.append(edits[0])
        edits = edits[1:]
    return out_edits

def get_edits_initial(source, target, edits):
    edits = get_edits(source, target, edits)
    wordspace_merged_edits = merge_ws(source, target, edits)
    return wordspace_merged_edits

def get_edits(source, target, edits):
    new_edits = []
    for edit in edits:
        op = edit[0]
        if op != "M":
            new_edits.append(edit)
    return new_edits

# mergeop: Merge all edits of the same operation type.
def get_edits_group_type(source, target, edits):
    new_edits = []
    for op, group in groupby(edits, lambda x: x[0]):
        if op != "M":
            new_edits.extend(merge_edits(list(group)))
    return new_edits

# allmerge: Merge all adjacent edits of any operation type, except M.
def get_edits_group_all(source, target, edits):
    new_edits = []
    for op, group in groupby(edits, lambda x: True if x[0] == "M" else False):
        if not op:
            new_edits.extend(merge_edits(list(group)))
    return new_edits

def korean_lemma_cost(A, B):
    """
>>> lemmatizer.lemmatize('하다')
[('하다', 'Adjective'), ('하다', 'Adjective'), ('하다', 'Verb'), ('하다', 'Verb')]
>>> lemmatizer.lemmatize('하는')
[('하다', 'Verb'), ('하다', 'Adjective'), ('하다', 'Verb'), ('하다', 'Adjective'), ('하다', 'Verb'), ('하다', 'Adjective')]
>>> lemmatizer.lemmatize('합니까')
[('하다', 'Adjective'), ('하다', 'Verb')]
    :param_A: "하다"
    :param_B: "하는"
    the first word match but the pos doesn't match.
    :return: 0.1 (out of 0, 0.1, or 0.499)
    """
    lemma_A = lemmatizer.lemmatize(A)
    lemma_B = lemmatizer.lemmatize(B)
    if len(lemma_A) == 0 or len(lemma_B) == 0:
        return 0.499 # one+ of them doesn't have lemma output
    else: # both of them have at least one lemma
        if lemma_A[0] == lemma_B[0]: # Lemma form and pos exactly match
            return 0
        elif lemma_A[0][0] == lemma_B[0][0]: # Lemma form match but pos doesn't match
            return 0.1
        return 0.499

def korean_pos_cost(A_pos, B_pos): # A_pos and B_pos is string(e.x. "VV", "NNG")
    try:
        A_set, B_set = set(A_pos), set(B_pos)
    except:
        print("ERROR: ", A_pos, B_pos)
        import pdb; pdb.set_trace()
    if A_set == B_set:
        return 0
    elif A_set.intersection(B_set) != set(): # 공통적으로 겹치는게 하나라도 있을때
        return 0.1
    elif A_set in KOREAN_CONTENT_POS and B_pos in KOREAN_CONTENT_POS:
        return 0.25
    else:
        return 0.5

def korean_char_cost(A, B):
    alignments = DL.WagnerFischer(A, B)
    alignment = next(alignments.alignments(True))
    return alignments.cost / float(len(alignment))

# A: 가니 B: 가다, A_pos: [VV, EFN], B_pos: [VCP]
def token_substitution_korean(A, B, A_pos, B_pos):

    cost = korean_lemma_cost(A, B) + korean_pos_cost(A_pos, B_pos) + korean_char_cost(A, B)
    return cost

# orig_str: A list of original sentence token strings.
# cor_str: A list of corrected sentence token strings.
# orig_extra: pos output of orig_str
# cor_extra: pos output of cor_str
# Input 7: The merging strategy you want to use to merge edits.
# Output: A list of lists. Each sublist is an edit of the form:
# edit = [orig_start, orig_end, cat, cor, cor_start, cor_end]
def getAutoAlignedEdits(orig_str, cor_str, orig_extra, cor_extra, merge_strategy, hobj, verbose=False, verbose_unclassified=False):
    alignments = DL.WagnerFischer(orig_str, cor_str, orig_extra, cor_extra, substitution=token_substitution_korean)
    # Get the first best alignment.
    alignment = next(alignments.alignments())
    # Convert the alignment into edits; choose merge strategy
    # Currently, we only do merge-equal strategy.
    edits = get_edits_initial(orig_str, cor_str, get_opcodes(alignment))
    proc_edits = []
    for edit in edits:
        code = edit[0]
        orig_start = edit[1]
        orig_end = edit[2]
        if code in SPECIAL_CODE:
            error_type = SPC2CODE[code]
        else:
            error_type = get_error_type(orig_str, cor_str, orig_extra, cor_extra, edit, hobj, code, verbose=verbose, verbose_unclassified=verbose_unclassified)
        cor_start = edit[3]
        cor_end = edit[4]
        cor_tok = " ".join(cor_str[cor_start:cor_end])
        proc_edits.append([orig_start, orig_end, error_type, cor_tok, cor_start, cor_end])
    return proc_edits

class ErrorAnalyzer():
    def __init__(self, orig_str, cor_str, orig_extra, cor_extra, edit, hobj, code, verbose=False, verbose_unclassified=False): # orig_extra and cor_extra should not be None!
        self.info = {
            'meta': {
                'orig_str': orig_str,
                'cor_str': cor_str,
                'orig_extra': orig_extra,
                'cor_extra': cor_extra
            },
            'original_token': orig_str[edit[1]:edit[2]],
            'original_pos': orig_extra[edit[1]:edit[2]],
            'corrected_token': cor_str[edit[3]:edit[4]],
            'corrected_pos': cor_extra[edit[3]:edit[4]],
            'original_modified_sequence': [edit[1], edit[2]],
            'corrected_modified_sequence': [edit[3], edit[4]],
            'edit': edit,
            'code': code,
            'verbose': verbose,
            'verbose_unclassified': verbose_unclassified,
        }
        self.pos = {
            'punct': set(['SF', 'SP', 'SS', 'SE', 'SO', 'SW']),
        }
        self.hobj = hobj

    def check_spell(self):
        if self.info['code'] == 'S' and (len(self.info['corrected_token']) == 1 and len(self.info['original_token']) == 1)\
                                and len(self.info['corrected_token'][0]) == len(self.info['original_token'][0]):  # Spelling error
            #cost = DL.WagnerFischer(self.info['original_token'], self.info['corrected_token'],
            #                        self.info['original_pos'], self.info['corrected_pos'],
            #                        substitution=token_substitution_korean).cost
            #cost = round(cost, 5)
            spell_list = self.hobj.suggest(self.info['original_token'][0])
            if spell_list == [] or (self.info['original_token'][0] not in spell_list and self.info['corrected_token'][0] in spell_list):
                return True
        return False

    def check_punct(self, all_pos_set):
        if len(self.pos['punct'].intersection(all_pos_set)) != 0:
            return True
        return False

    def flatten_pos(self, pos_list):
        pos_set = set()
        for phrase in pos_list:
            pos_set = pos_set.union(set([x[1] for x in phrase]))
        return pos_set

    def classify_output(self):
        """
        example of self.info:
        > Substitution: {
        'original_token': ['소풍'],
        'original_pos': [[('소풍', 'NNG')]],
        'corrected_token': ['소풍을'],
        'corrected_pos': [[('소풍', 'NNG'), ('을', 'JKO')]],
        'original_modified_sequence': [3, 4],
        'corrected_modified_sequence: [3, 4]
        }
        > Insertion: {
        'original_token': [],
        'original_pos': [],
        'corrected_token': ['다시'],
        'corrected_pos': [[('다시', 'MAG')]],
        'original_modified_sequence': [10, 10],
        'corrected_modified_sequence': [10, 11]
        }
        > Deletion
        {'code': 'I',
        'corrected_modified_sequence': [4, 5],
        'corrected_pos': [[('모델', 'NNG'), ('이', 'VCP'), ('다', 'EFN')]],
        'corrected_token': [''],
        'edit': ('I', 4, 4, 4, 5),
        'meta': {'cor_extra': [[('차이나', 'NNG'), ('유니', 'NNP')],
                                [('콤', 'UN')],
                                [('전용', 'NNG')],
                                [('듀얼', 'NNG'), ('심', 'NNG')],
                                [('모델', 'NNG'), ('이', 'VCP'), ('다', 'EFN')]],
                'cor_str': ['차이나유니콤', '전용', '듀얼심', '모델이다', ''],
                'orig_extra': [[('차이나', 'NNG'), ('유니', 'NNP'), ('콤', 'UN')],
                                [('전용', 'NNG')],
                                [('듀얼', 'NNG'), ('심', 'NNG')],
                                [('모델', 'NNG')]],
                'orig_str': ['차이나유니콤', '전용', '듀얼심', '모델']},
        'original_modified_sequence': [4, 4],
        'original_pos': [],
        'original_token': [],
        'verbose': True,
        'verbose_unclassified': False}
        """
        #original_pos = self.flatten_pos(self.info['original_pos'])
        #changed_pos_set = self.flatten_pos(self.info['corrected_pos'])
        #all_pos_set = original_pos.union(changed_pos_set)
        #if self.info['verbose']:
        #    print(f"{self.info['original_pos']} => {self.info['corrected_pos']}")
        # first, filter out D / I errors
        if self.info['code'] == 'D':
            return 'DELETION'
        if self.info['code'] == 'I':
            return 'INSERTION'
        # check if spelling error
        if self.check_spell():
            return 'SPELL' # + str(cost)
        try:
        # check if 축약형
            if len(self.info['original_pos']) == 1 and len(self.info['corrected_pos']) == 1 \
            and "".join([x[0] for x in self.info['original_pos'][0]]) == "".join([x[0] for x in self.info['corrected_pos'][0]]):
                return 'SHORTEN'
            elif len(self.info['original_pos']) <= 1 and len(self.info['corrected_pos']) <= 1:
                original_pos = self.info['original_pos'][0] if len(self.info['original_pos']) == 1 else []
                corrected_pos = self.info['corrected_pos'][0] if len(self.info['corrected_pos']) == 1 else []
                alignment = next(DL.WagnerFischer(
                    [m[0] for m in original_pos], [m[0] for m in corrected_pos],
                    [m[1] for m in original_pos], [m[1] for m in corrected_pos]).alignments())
                edits = get_edits(original_pos, corrected_pos, get_opcodes(alignment))
                error_types = []
                for edit in edits:
                    code = edit[0]
                    if code in SPECIAL_CODE:
                        error_types.append(SPC2CODE[code])
                    elif code == 'D':
                      error_types += [adjust_pos_granularity(m[1]) for m in original_pos[edit[1]:edit[2]]]
                    else:
                        error_types += [adjust_pos_granularity(m[1]) for m in corrected_pos[edit[3]:edit[4]]]
                error_types = sorted(list(set(error_types)))
                if len(error_types) == 1:
                    return error_types[0]
                error_types = [PARTICLE if e.startswith(PARTICLE) else e for e in error_types]
                error_types = sorted(list(set(error_types)))
                if len(error_types) == 1:
                    return error_types[0]
                composite_error_types = composite_error_type(error_types)
                if composite_error_types is not None:
                    return composite_error_types

                if NOUN in error_types:
                    if VERB not in error_types and ADJECTIVE not in error_types:
                        return UNCLASSIFIED #NOUN + ":OTHER"
                elif VERB in error_types:
                    if ADJECTIVE not in error_types:
                        return CONJUGATION
                elif ADJECTIVE  in error_types:
                    return CONJUGATION

                if self.info['verbose_unclassified']:
                    return UNCLASSIFIED + ":,".join(error_types)
        except:
            import pdb; pdb.set_trace()
        return UNCLASSIFIED


"""
Input example (from edit_extraction code)

orig_str
['사람들이', '꽃을', '보고', '소풍', '가요', '.']
cor_str
['사람들이', '꽃을', '보고', '소풍을', '가요', '.']
orig_extra
[[('사람', 'NNG'), ('들', 'XSN'), ('이', 'VCP')], [('꽃', 'NNG'), ('을', 'JKO')], [('보', 'VV'), ('고', 'ECE')], [('소풍', 'NNG')], [('가요', 'NNG')], [('.', 'SF')]]
cor_extra
[[('사람', 'NNG'), ('들', 'XSN'), ('이', 'VCP')], [('꽃', 'NNG'), ('을', 'JKO')], [('보', 'VV'), ('고', 'ECE')], [('소풍', 'NNG'), ('을', 'JKO')], [('가요', 'NNG')], [('.', 'SF')]]
edit
('X', 3, 4, 3, 4)
"""


def get_error_type(orig_str, cor_str, orig_extra, cor_extra, edit, hobj, code, verbose=False, verbose_unclassified=False):
    analyzer = ErrorAnalyzer(orig_str, cor_str, orig_extra, cor_extra, edit, hobj, code, verbose=verbose, verbose_unclassified=verbose_unclassified)
    return analyzer.classify_output()

