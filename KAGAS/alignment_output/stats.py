
def file(filename):
    with open(filename, 'r+') as fin:
        data = fin.read()
    data = [x for x in data.split('\n') if x != '']
    ho = [int(x.split('\t')[1]) for x in data]
    tot = sum(ho)
    print(filename, tot)
    import pdb; pdb.set_trace()

file('korean_learner_full_stats.csv')
file('lang8_full_stats.csv')
file('native_full_stats.csv')
