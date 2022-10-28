# Korean Spellchecker by [hunspell](https://github.com/spellcheck-ko/hunspell-dict-ko)
([Description](https://www.slideshare.net/changwoo/hunspell-works))
## Installation
### Mac OS
Naive build fail; please install hunspell by referencing [this blog post](http://pankdm.github.io/hunspell.html)
and [github
issue](https://github.com/blatinier/pyhunspell/issues/33#issuecomment-332636904).
For me, I did the following:
```
brew install hunspell
hunspell -D # Add ko.aff and ko.dic inside here, and see if hunspell loads this correctly 
ln -s /usr/local/Cellar/hunspell/1.7.0_2/lib/libhunspell-1.7.0.dylib /usr/local/Cellar/hunspell/1.7.0_2/lib/libhunspell.dylib
CFLAGS=$(pkg-config --cflags hunspell) LDFLAGS=$(pkg-config --libs hunspell) pip install hunspell
```

### Linux
You should first install prerequired dependencies
```
pip install python-dev-tools
apt-get update -y
apt-get install -y libhunspell-dev
```
After than, you can successfully install hunspell
```
pip install hunspell
```


## Usage Example
```
python3
>>> import hunspell
>>> hobj = hunspell.HunSpell('/Users/soyoung/Library/Spelling/ko.dic', '/Users/soyoung/Library/Spelling/ko.aff')
>>> hobj.suggest("안뇽")
['안녕']
>>> hobj.suggest("너무해오")
['너무하오', '너무해요', '너무해야', '너무해도', '너 무해오', '너무 해오', '너무해 오', '너무해온', '너무해올', '너무해옴', '너무해와']
>>> # have potential to seperate two words correctly, but not useful for 2+
words
>>> hobj.suggest("두글자")
['둥글자', '두 글자']
>>> hobj.suggest("이건대단해")
['이건 대단해', '건들대다']
>>> hobj.suggest("무슨원리일까")
['무슨 원리일까', '원심분리기일까']
>>> hobj.suggest("복사해사용")
['복사해 사용', '복사하다']
>>> hobj.suggest("복사해사용할수도있다")
[]
```

