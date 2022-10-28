Korean Learner to M2 - pre-processor
------------------

This repository is based on [M2 scorer](https://github.com/snukky/wikiedits) and [korean learner corpus](https://kcorpus.korean.go.kr/index/goBoardView.do?boardSeq=66)
The modified NIKL corpus (Korean Learner Corpus) is available only for `non-commercial` purposes.

Requirements
------------

This package is tested with Python 3.7.

Required python packages:

- `xmltodict`
- `numpy`

Usage
-----

### prepare data

```shell
gzip ./data/korean_learner_corpus_error_sentences.xml.gz
```

### script options

```shell
$python learner_processor.py -h
usage: learner_processor.py [-h] XML ORI COR COM M2

한국어교수학습샘터 한국어 학습자 오류 교정 자료 전처리용 스크립트

positional arguments:
  XML         merged korean learner xml file. check data dir
  ORI         filename to write original sentences
  COR         filename to write corrected sentences
  COM         filename to write original and corrected sentence combined
  M2          filename to write m2

optional arguments:
  -h, --help  show this help message and exit
```

usage example:
```shell
python3 learner_processor.py
/Users/soyoung/Desktop/korean_learner_corpus_error_sentences.xml ./ori ./cor
./com ./m2
```
