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
#!/usr/bin/env python

# Courtney Napoles
# <courtneyn@jhu.edu>
# 21 June 2015
# ##
# compute_gleu
#
# This script calls gleu.py to calculate the GLEU score of a sentence, as
# described in our ACL 2015 paper, Ground Truth for Grammatical Error
# Correction Metrics by Courtney Napoles, Keisuke Sakaguchi, Matt Post,
# and Joel Tetreault.
#
# For instructions on how to get the GLEU score, call "compute_gleu -h"
#
# Updated 2 May 2016: This is an updated version of GLEU that has been
# modified to handle multiple references more fairly.
#
# This script was adapted from compute-bleu by Adam Lopez.
# <https://github.com/alopez/en600.468/blob/master/reranker/>

import argparse
import sys
import os
from .gleu import GLEU
import scipy.stats
import numpy as np
import random


def get_gleu_stats(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    res = ['%f' % mean, '%f' % std]
    if len(scores) != 1:
        ci = scipy.stats.norm.interval(0.95, loc=mean, scale=std)
        res.append('(%.3f,%.3f)' % (ci[0], ci[1]))
    return res


def run_gleu(reference='', source='', hypothesis=''):
    args = {
        'n': 4,
        'l': 0.0,
        'reference': reference,
        'source': source,
        'hypothesis': hypothesis,
        'debug': False
    }

    # if there is only one reference, just do one iteration, which is always our case.
    num_iterations = 1

    gleu_calculator = GLEU(args['n'])

    gleu_calculator.load_sources(args['source'])
    gleu_calculator.load_references([args['reference']])

    for hpath in [args['hypothesis']]:
        instream = open(hpath)
        hyp = [line.split() for line in instream]

        if args['debug']:
            print(os.path.basename(hpath))

        # first generate a random list of indices, using a different seed
        # for each iteration
        indices = []
        for j in range(num_iterations):
            random.seed(j * 101)
            indices.append([random.randint(0, 0)
                            for _ in range(len(hyp))])

        if args['debug']:
            print('')
            print('===== Sentence-level scores =====')
            print('SID Mean Stdev 95%CI GLEU')

        iter_stats = [[0 for i in range(2 * args['n'] + 2)]
                      for j in range(num_iterations)]

        for i, h in enumerate(hyp):

            gleu_calculator.load_hypothesis_sentence(h)
            # we are going to store the score of this sentence for each ref
            # so we don't have to recalculate them 500 times

            stats_by_ref = [None for r in range(1)]

            for j in range(num_iterations):
                ref = indices[j][i]
                this_stats = stats_by_ref[ref]

                if this_stats is None:
                    this_stats = [s for s in gleu_calculator.gleu_stats(
                        i, r_ind=ref)]
                    stats_by_ref[ref] = this_stats

                iter_stats[j] = [sum(scores)
                                 for scores in zip(iter_stats[j], this_stats)]

            if args['debug']:
                # sentence-level GLEU is the mean GLEU of the hypothesis
                # compared to each reference
                for r in range(1):
                    if stats_by_ref[r] is None:
                        stats_by_ref[r] = [s for s in gleu_calculator.gleu_stats(
                            i, r_ind=r)]

                print(i)
                print(' '.join(get_gleu_stats([gleu_calculator.gleu(stats, smooth=True)
                                               for stats in stats_by_ref])))

        if args['debug']:
            print('\n==== Overall score =====')
            print('Mean Stdev 95%CI GLEU')
            print(' '.join(get_gleu_stats([gleu_calculator.gleu(stats)
                                           for stats in iter_stats])))
        else:
            return (get_gleu_stats([gleu_calculator.gleu(stats)
                                  for stats in iter_stats])[0])

