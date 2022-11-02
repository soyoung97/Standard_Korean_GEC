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
import argparse
from collections import OrderedDict
import xml.etree.ElementTree as xmlparser
import json

import xmltodict

from hangle_util import check_jaum_token, merge_two_token, is_jaum, apply_short_rule

'''
<SENTENCE from="231" to="269">
        <s>그리고 티터 씨는 항상 귀엽게 옷어서 같이 공부할 때 기분이 좋다. </s>
        <Privacy_Name from="235" to="237">티터</Privacy_Name>
        <MorphemeAnnotations>
          <word>
            <w>그리고</w>
            <morph analyzedType="Ambiguity" from="231" pos="MAJ" subsequence="1" to="234">그리고</morph>
          </word>
          <word>
            <w>티터</w>
            <morph analyzedType="Fail" from="235" pos="NNP" subsequence="1" to="237">티터</morph>
          </word>
          <word>
            <w>씨는</w>
            <morph analyzedType="Ambiguity" from="238" pos="NNB" subsequence="1" to="240">씨</morph>
            <morph analyzedType="Ambiguity" from="238" pos="JX" subsequence="2" to="240">는</morph>
          </word>
          <word>
            <w>항상</w>
            <morph analyzedType="Normal" from="241" pos="MAG" subsequence="1" to="243">항상</morph>
          </word>
          <word>
            <w>귀엽게</w>
            <morph analyzedType="Normal" from="244" pos="VA" subsequence="1" to="247">귀엽</morph>
            <morph analyzedType="Normal" from="244" pos="EC" subsequence="2" to="247">게</morph>
          </word>
          <word>
            <w>옷어서</w>
            <morph analyzedType="Manual" from="248" pos="VV" subsequence="1" to="251">옷</morph>
            <morph analyzedType="Manual" from="248" pos="EC" subsequence="2" to="251">어서</morph>
          </word>
          <word>
            <w>같이</w>
            <morph analyzedType="Normal" from="252" pos="MAG" subsequence="1" to="254">같이</morph>
          </word>
          <word>
            <w>공부할</w>
            <morph analyzedType="Normal" from="255" pos="NNG" subsequence="1" to="258">공부</morph>
            <morph analyzedType="Normal" from="255" pos="XSV" subsequence="2" to="258">하</morph>
            <morph analyzedType="Normal" from="255" pos="ETM" subsequence="3" to="258">ㄹ</morph>
          </word>
          <word>
            <w>때</w>
            <morph analyzedType="Ambiguity" from="259" pos="NNG" subsequence="1" to="260">때</morph>
          </word>
          <word>
            <w>기분이</w>
            <morph analyzedType="Ambiguity" from="261" pos="NNG" subsequence="1" to="264">기분</morph>
            <morph analyzedType="Ambiguity" from="261" pos="JKS" subsequence="2" to="264">이</morph>
          </word>
          <word>
            <w>좋다.</w>
            <morph analyzedType="Ambiguity" from="265" pos="VA" subsequence="1" to="268">좋</morph>
            <morph analyzedType="Ambiguity" from="265" pos="EF" subsequence="2" to="268">다</morph>
            <morph analyzedType="Normal" from="265" pos="SF" subsequence="3" to="268">.</morph>
          </word>
        </MorphemeAnnotations>
        <LearnerErrorAnnotations>
          <word>
            <w>옷어서</w>
            <morph from="248" subsequence="1" to="251" wordStart="Start">
              <Proofread pos="VV">웃</Proofread>
              <ErrorArea type="CVV" />
              <ErrorPattern type="MIF" />
            </morph>
            <morph from="248" subsequence="2" to="251" wordStart="None">
              <Preserved>어서</Preserved>
            </morph>
          </word>
        </LearnerErrorAnnotations>
      </SENTENCE>

'''


'''
ErrorPattern : 오류양상
ErrorArea : 오류위치
ErrorLevel : 오류층위 
'''

'''
look up https://github.com/chrisjbryant/errant > cat_rules.py
each correct format
ex) A 2 3|||R:VERB:FORM|||see|||REQUIRED|||-NONE-|||0
1. 'A' means correction
2. number span : calc based on space
3. R:VERB:FORM 
 3-1. correction type. 
    1) UNK : unknown 
    2) M : Missing
    3) U : Unnecessary
    4) R : Replacement
 3-2. edit type
    1) "NOUN:POSS"
        # Possessive noun suffixes; e.g. ' -> 's
    2) "CONTR"			
        # Contraction. Rule must come after possessive.
    3) "VERB:FORM"
        # Infinitival "to" is treated as part of a verb form.
    4) "VERB:TENSE"
        # Auxiliary verbs.
        # if set(dep_list).issubset({"aux", "auxpass"}):
    5) "VERB"
        # To-infinitives and phrasal verbs.
        if set(pos_list) == {"PART", "VERB"}:
    6) some position based tag. use Spacy tag
    7) "OTHER"	
        # Tricky cases
'''

def sentence_process(sent_dict):
    '''

    :param sent_dict : ordered dict parsed from dict
    :return ordered dict
    - add content
    1) corrected setnece
    2) M2 format Annotation
    '''
    sent_dict = sent_dict['SENTENCE']

    cor_sent = []
    m2_edits = []

    orig_sent = sent_dict['s']+'\n'
    print('orig >>>', orig_sent)
    ano_sent_t = "A {} {}|||{}|||{}|||REQUIRED|||-NONE-|||0\n"

    #1. merge corrected sentence
    sent_morphs = sent_dict['MorphemeAnnotations']['word']
    if not isinstance(sent_morphs, list):
        sent_morphs = [sent_morphs]

    for i, word in enumerate(sent_morphs) :

        tok = word['w']
        print("현재 {} , tok >>> {}".format(i,tok))

        edit_cnt = len(m2_edits)
        proof_cnt = len(cor_sent)

        l = sent_dict['LearnerErrorAnnotations']['word']

        if not isinstance(l, list):
            l = [l]

        for cor_word in l : #

            if cor_word['w'] == tok : # match with same original word on LearnerErrorAnnotations and MorphemAnnotations
                morph = cor_word.get('morph')

                if isinstance(morph, list):
                    proof_list = []
                    local_edits = []
                    for m in morph :
                        ano, proof = get_ano_and_cor(m)
                        print("n morph >>> ",ano,proof)
                        if proof:
                            proof_list.append(proof)
                        if ano :
                            local_edits.append(ano)

                    # check_jaum_token('가리키ㄴ다') => 가리킨다
                    # apply_short_rule('하였다') => 했다

                    proof = apply_short_rule(check_jaum_token(''.join(proof_list)))
                    if proof :
                        cor_sent.append(proof)
                        m2_edits.append(ano_sent_t.format(i, i + 1, '::'.join(local_edits), ''.join(proof)))
                    else :
                        m2_edits.append(ano_sent_t.format(i, i + 1, '::'.join(local_edits), ''.join(proof_list)))


                else :
                    ano,proof = get_ano_and_cor(morph)
                    proof = apply_short_rule(check_jaum_token(proof))
                    print("1 morph >>> ",ano, proof)
                    if proof :
                        cor_sent.append(proof)
                    if ano :
                        m2_edits.append(ano_sent_t.format(i, i + 1, ano, proof))

        if len(m2_edits) == edit_cnt and len(cor_sent) == proof_cnt:
            cor_sent.append(tok)

    print(len(cor_sent), cor_sent)

    for i in range(len(cor_sent)-1, 0, -1):
        if is_jaum(cor_sent[i][0]):
            a,b = merge_two_token(cor_sent[i-1], cor_sent[i])
            cor_sent[i-1] = a
            cor_sent[i]= None

    corrected = ' '.join(x for x in cor_sent if x)+'\n'
    print(corrected)
    print(m2_edits)

    return orig_sent, corrected, m2_edits


# return Error Type and Error Annotation. ex: ('J', '를')
def get_ano_and_cor(morph):
    print(f'get_ano_and_cor >>> {morph}')
    if morph.get('Preserved') :
        return '', morph.get('Preserved')

    elif morph.get('Removed'):
        return None, None

    else :
        proof_read = morph.get('Proofread')

        if isinstance(proof_read, OrderedDict):
            proof_read = proof_read.get('#text')
            proof_read_type = morph.get('Proofread').get('@pos')

            if proof_read == 'ADD':

                return proof_read_type[:1], None

        elif not proof_read :
            return None, None

        elif proof_read == 'ADD':
            proof_read_type = 'UNK'
            return proof_read_type, None

        else :
            proof_read_type = 'UNK'
            return proof_read_type, proof_read


    print(f'get_ano_and_cor return >>> type:{proof_read_type}, read:{proof_read}')

    return proof_read_type if proof_read_type == 'UNK' else proof_read_type[:1], proof_read if proof_read else None



def make_learner_xml(path, filename="korean_learner_corpus_error_sentences.xml"):
    all = []
    for filename in os.listdir(path):
        all.extend(parse_xml(path + '/' + filename))
    root_node = xmlparser.Element('root')
    [root_node.append(e) for e in all]
    xmlparser.ElementTree(root_node).write(filename, encoding='utf-8')

def parse_xml(file):
    tree = xmlparser.parse(file)
    root = tree.getroot()
    lists = []
    for s in root.iter('SENTENCE'):
        # print("attrib : {} \t text : {} ".format(s.attrib,s.text))
        err_ano = s.find('LearnerErrorAnnotations')
        if err_ano:
            if len(err_ano) >0 :
              lists.append(s)

    return lists

def gather_xml(xml_dir="."):
    root_path = os.path.abspath(xml_dir)
    file_base_path = os.path.abspath(".")

    for path, name, files in os.walk(root_path):
        if files and '문어' in path.split('/')[-1]:
            for file in files:
                if 'xml' in file:
                    cur_path = path + '/' + file
                    dest_path = file_base_path + path.split('/')[-1] + '.xml'
                    os.rename(cur_path, dest_path)

def parse_main_args():

    parser = argparse.ArgumentParser(
        description="한국어교수학습샘터 한국어 학습자 오류 교정 자료 전처리용 스크립트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("xml_file_path", metavar='XML', default="./data/korean_learner_corpus_error_sentences.xml",
                        help="merged korean learner xml file. check data dir")

    parser.add_argument("original", metavar="ORI", default="original.txt", help="filename to write original sentences")
    parser.add_argument("corrected", metavar="COR", default="corrected.txt", help="filename to write corrected sentences")
    parser.add_argument("combined",  metavar="COM", default="combined.txt", help="filename to write original and corrected sentence combined")
    parser.add_argument("m2",  metavar="M2", default="korean_learner.m2", help="filename to write m2")

    args = parser.parse_args()
    return args


def main():
    args = parse_main_args()

    xml_path = os.path.abspath(args.xml_file_path)
    ori_file = args.original
    cor_file = args.corrected
    combined_file = args.combined
    m2_file = args.m2

    print(f"xml_path:{xml_path}\nori_file:{ori_file}\ncor_file:{cor_file}\ncombined_file:{combined_file}\nm2_file:{m2_file}")

    tree = xmlparser.parse(xml_path)

    with open(ori_file, 'w', encoding='utf-8') as o, open(cor_file, 'w', encoding='utf-8') as c, open(m2_file, 'w', encoding='utf-8') as m , open(combined_file, 'w', encoding='utf-8') as comb:
        root = tree.getroot()
        for node in root.iter('SENTENCE'):
            d = xmltodict.parse(xmlparser.tostring(node, encoding='utf-8'))
            # print(d)
            orig_sent, corrected, m2_edits = sentence_process(d)
            comb.writelines(orig_sent[:-1]+'\t'+corrected)
            o.writelines(orig_sent)
            c.writelines(corrected)
            m.write('S '+orig_sent+'\n')
            m.writelines(m2_edits)
            m.write('\n\n')

if __name__ == '__main__':
    main()
