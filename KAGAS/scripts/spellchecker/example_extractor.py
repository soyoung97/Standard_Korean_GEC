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

import hunspell

# Define arguments
output_name = 'lang8'
file_path = f'/Users/soyoung/Desktop/coding/research/GEC/GEC-Korean/src/errant/edit-extraction/scripts/spellchecker/outputs/{output_name}.txt'


hobj = hunspell.HunSpell('/Users/soyoung/Library/Spelling/ko.dic', '/Users/soyoung/Library/Spelling/ko.aff')

with open(file_path, 'r') as f:
    raw = f.read().split("\n")[:-1]

output = ""
for i, text in enumerate(raw):
    res = text.split("\t")
    src, tgt = res[0], res[1]
    words = src.split(" ")[:-1]
    suggestions = [hobj.suggest(w) for w in words]
    header = f"""번호: {i}
src: {src}
tgt: {tgt}
suggestion list:"""
    print(header)
    output += header
    for w, sug in zip(words, suggestions):
        output += f"\n{w : <15}: {sug}"
    output += "\n\n"

with open(f"example_output/{output_name}.txt", "w") as f:
    f.write(output)


