import os

conv = {'.txt': "withOUT punct split", "_punct.txt": "WITH punct split"}
#/output/korean_learner/generation/epoch0/val
for data in ['korean_learner', 'lang8', 'native', 'union']:
    for filetype in ['.txt', "_punct.txt"]:
        script = f"./m2scorer genout/{data}{filetype} /home/GEC-Korean/extract_data/{data}/{data}_val.m2"
        print(script)
        print(f"{data}: {conv[filetype]}")
        os.system(script)

