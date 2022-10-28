This is the official github repository for the paper: [towards standardizing Korean Grammatical Error Correction: Datasets and Annotation](https://arxiv.org/abs/2210.14389)

Code maintained by: [Soyoung Yoon](https://soyoung97.github.io/profile/)

For running the code, consider working on inside the docker image: `msyoon8/korean_gec:minimal` which is uploaded in docker hub.
The command I used is below:
```
docker run -it -d --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm --ipc=host --gpus all -v msyoon8/korean_gec:minimal /bin/bash
docker exec -it [docker container name] \bin\bash # work on it interactively inside
```
`msyoon8/korean_gec:minimal` has minimal settings - you need to clone this repository at /home and run `pip3 install -r requirements.txt`

# 0.Install dependencies
```
pip3 install -r requirements.txt
```
Short summary:

You first need to run the above commands. Else, we recommend using the provided docker image.
To get the datasets, follow the instructions below.
To run KAGAS, go to `KAGAS` and run the following sample code to see the result:
```
python3 parallel_to_m2_korean.py -orig sample_test_data/orig.txt -cor sample_test_data/corrected.txt -out sample_test_data/output.m2 -hunspell ./aff-dic
```

# 1. How to get data
## Kor-Lang8
1. Download raw lang8 file
    - Fill out [this form](https://docs.google.com/forms/d/17gZZsC_rnaACMXmPiab3kjqBEtRHPMz0UG9Dk-x_F0k/) which is from [this download page](https://sites.google.com/site/naistlang8corpora/) from lang-8 learner corpora (raw format).
    - From your email, download & unzip `lang-8-20111007-2.0.zip` (For mac, it is just `unzip lang-8-20111007-2.0.zip`)
    - Set the environment variable `PATH_TO_EXTRACTED_DATA` as the path for `lang-8-20111007-L1-v2.dat`. (ex: `PATH_TO_EXTRACTED_DATA=./lang-8-20111007-L1-v2.dat`)

2. Get raw korean sentences and filter them by using our code
```
cd get_data
git clone https://github.com/tomo-wb/Lang8-NAIST-extractor.git
cd Lang8-NAIST-extractor/scripts
python extract_err-cor-pair.py -d $PATH_TO_EXTRACTED_DATA -l2 Korean > ../../lang8_raw.txt
cd ../../
git clone https://github.com/SKTBrain/KoBERT.git
cd KoBERT
pip install .
cd ..
python filter.py
```
after running this code, you will get `kor_lang8.txt` on the current directory. (`.\get_data`)
Full logs after running filter.py is as follows:
```
1. Load data
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 406274/406274 [00:00<00:00, 1217655.24it/s]
dataset length 406271
2. Remove white spaces, misfunctioned cases, and other languages
dataset length 323531
3. Drop rows of other languages
4. Calculate variables and apply conditions
using cached model. /home/GEC_Korean_Public/get_data/.cache/kobert_news_wiki_ko_cased-1087f8699e.spiece
dataset length 203604
5. Save cleaned up data
6187
lang8 counter: 192183
unique lang8 data: 192183
Length before levenshtein:  122641
Length after levenshtein:  13080
Done!
```

## Kor-Learner and Kor-Native
**IMPORTANT** The Kor-Learner corpus is made from the NIKL corpus, and it is only allowed to
use & distribute under non-commercial purposes. Please refer to the [original agreement form](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)
The Kor-Native corpus is made from the (1) The Center for Teaching and Learning for Korean, and (2) National Institute of Korean Language, and
it's also only allowed to use under non-commercial purposes.
You can get the downloadable link for the dataset after filling in [this form](https://forms.gle/kF9pvJbLGvnh8ZnQ6) (You need to log in to google)
We recommend you download the data and put it under `get_data` directory.

For detailed code about building Kor-Learner, please refer to the directory: `get_data/korean_learner`([original private
repository](https://github.com/kimgyu/korean-learner).

# 2. How to run the models
**Disclaimer**: The original model code was implemented years ago,
so we implemented a new, easier one that is more easily accessible.

We are also planning to open an easy-accessible version of the pre-trained model by our dataset.
To reproduce the original experiments, please refer to [this spreadsheet](https://docs.google.com/spreadsheets/d/1II_BB10YPijp1Rgw3ZgQElvv6pw7xINOdTpJbAPz484/edit?usp=sharing).
It contains all the information about model checkpoints(link to google drive), command logs, and so on.

If you want to reproduce it youserlf, look at the codes under `src/KoBART_GEC`. Please use the docker image `msyoon8/default:gec` to run the original code.

Kobart transformers are referenced by [this repository](https://github.com/hyunwoongko/kobart-transformers).
## Prepare dataset
### Split into train/test/val, Making union files
- It would be helpful if you run the `get_data/process_for_train.py` code.
- Please use the same code to make union files and split them into train,test, and val.
```
>>> python3 process_for_training.py
Select actions from the following: (punct_split, merge_wiki, split_pairs, train_split, q, merge_files, reorder_m2, make_sample, make_union) : train_split
Please specify the whole pair data path: /home/GEC_Korean_Public/get_data/native.txt
Length = train: 12292, val: 2634, test: 2634, write complete
Select actions from the following: (punct_split, merge_wiki, split_pairs, train_split, q, merge_files, reorder_m2, make_sample, make_union) : q
Bye!
```

## Train
Outputs will be saved at `outputs/` directory.
Sample code is listed below.
```
make native
# Or, type full commands:
# python3 run.py --data native --default_root_dir ../../output --max_epochs 10 --lr 3e-05 --SEED 0 --train_file_path ../../get_data/native_train.txt --valid_file_path ../../get_data/native_val.txt --test_file_path ../../get_data/native_test.txt
```

## Evaluation
In order to evaluate on m2 file, you need to have the m2 file of format `{data}_{mode}.m2` (e.g. native\_val.m2) at `get_data`.
You should run KAGAS first on the data to make corresponding m2 file.
For the gleu module, we modified the [official code](https://github.com/cnap/gec-ranking) to enable import & work on python3. It is at `src/eval`.
For m2scorer, we used the [python3 compatible version of m2scorer](https://github.com/ayaka14732/m2scorer.git).
Please note that running the m2 scorer may take a really long time (hours).
Evaluation is automatically done for each epoch once you run the training code.


# 3. How to run KAGAS

Based on the original [ERRANT repository](https://github.com/chrisjbryant/edit-extraction) 
We modified some parts of the code to work for Korean. 

# Code structure

```
|__ alignment_output/ Output examples from
        |__ ALL/ --> WS & PUNCT & SPELL edits
        |__ allMerge/ --> Allmerge version of alignment
        |__ allSplit/ --> AllSplit version of alignment
        |__pos/ --> pos splitted sample log by kkma
|__scripts/
        |__ extractPos/ --> The original implemented version by @JunheeCho - modules which extract POS information from sentences
        |__ spellchecker/ --> Contains example script that runs the korean hunspell and the outputs.
        |__ align_text.py --> ERRANT version of align_text.py that has English version of linguistic cost.
        |__ align_text_korean.py --> [IMPORTANT] korean version of align_text.py. Main function where it classify/align/merge Edits
        |__ rdlextra.py --> Implementation of Damerau-Levenshtein Distance. (Adapted from the initial version of ERRANT)
|__Makefile --> simple script that helps automatically generate output into alignment_output. Most of times type "make debug"
|__parallel_to_m2.py --> Original alignment code from ERRANT.
|__parallel_to_m2_korean.py --> [IMPORTANT] korean version of parallel_to_m2.py. Starting function. Process files and outputs m2 file.
|__sample_test_data/ --> simple test file that you can give as input to parallel_to_m2 or parallel_to_m2_korean.py. Contains original and corrected sample files.
|__ws_error_dist.py --> Code that is used to generate the distribution of word space errors by kkma.
|__ws_error/ --> Sample outputs from ws_error_dist.py
```

## Starting point
All scripts run properly when you run it currently in **this directory**(KAGAS/)
The main starting point of function is `parallel_to_m2_korean.py`. For Korean, we mainly modify `parallel_to_m2_korean.py` and `align_text_korean.py`.

## Sample running code
```
python3 parallel_to_m2_korean.py -orig sample_test_data/orig.txt -cor sample_test_data/corrected.txt -out sample_test_data/output.m2 -hunspell ./aff-dic
```
Additional explanations:
- If you add `-save`, it means that m2 output files will be saved at -out path. Otherwise, it will just be printed out.
- If you add `-logset`, it means that you want to only log(print) specific types.
You can put multiple logset types e.g. `-logset WS SPELL PUNCT`. If you omit `-logset`, it returns all possible cases.

## Setup

### Korean Dictionary
From this [github release](https://github.com/spellcheck-ko/hunspell-dict-ko/releases),
Korean dictionary [ko-aff and ko-dic](https://github.com/spellcheck-ko/hunspell-dict-ko/releases/download/0.7.92/ko-aff-dic-0.7.92.zip) are downloadable.
You should put them into `src/errant/edit-extraction/aff-dic/`, and save them as `ko.aff` and `ko.dic`.
In our repository, we already downloaded hunspell dictionary and put them inside `/KAGAS/aff-dic/` directory, so you don't need to additionally download them.
when you run KAGAS, give the path as the following example:
```
python3 parallel_to_m2_korean.py ..... -hunspell $DIRECTORY_WHERE_HUNSPELL_LIBRARIES_ARE_SAVED ...
```
For our case, the code would be:
```
python3 parallel_to_m2_korean.py ..... -hunspell ./aff-dic...
```
### Install Hunspell
- For MacOS, follow the directions from [this blog post](https://pankdm.github.io/hunspell.html)
- For linux/others, it is highly recommended to use the [Cython wrapper on Hunspell dictionary](https://github.com/MSeal/cython_hunspell),
since installation is not easy. For the wrapper version, you just need to install the following:
```
pip3 install cyhunspell
```
It would have already been installed if you previously ran `requirements.txt`.
### KoNLPy
You may need to install java (if not already) to run KKma. You can try

```
apt-get install openjdk-8-jdk
```

## Where to modify, if you want to customize?
### Alignments
For modifying alignment costs, you should look at `getAutoAlignedEdits` and `token_substitution_korean` from `scripts/align_text_korean.py`.

### Edit classification
If you want to classify currently "NA" typed edits, you should edit `classify_output()` from class `ErrorAnalyzer()` located in `scripts/align_text_korean.py`.
Currently, all error types except "WS"(Word Space) are classified here. Since word space edit needs merging, it is
specially handled beforehand at `merge_ws` from `scripts/align_text_korean.py`.

# Contact

If you have any questions, suggestions or bug reports, you can contact the authors at:
soyoungyoon at kaist.ac.kr
