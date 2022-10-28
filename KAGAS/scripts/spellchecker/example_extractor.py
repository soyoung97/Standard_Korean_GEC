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


