Python code based on the [original GLEU repository](https://github.com/cnap/gec-ranking).

## Difference from original code
- Adjusted & Modified from python2 to python3.
    - Change xrange to range
    - Add `()` after print function
- Fix minor errors related to calculating invalid multiply value in `scipy.stats.norm.interval` (line 35 at run_gleu.py)
- Add gleumodule.py which is a modified version of run_gleu.py that can be used during runtime at other python script.

## Example reference code
```
python3 run_gleu.py -r reference.txt -s source.txt -o hypothesis.txt
hypothesis.txt
0.872146

>>> from gleumodule import run_gleu
>>> run_gleu(reference='reference.txt', source='source.txt', hypothesis='hypothesis.txt')
'0.872146'
```
The default settings are: ` -n 4 -l 0.0`

Also works well with Korean text.