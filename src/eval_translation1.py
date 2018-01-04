#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

# Copyright (C) 2016-2017  Mikel Artetxe <artetxem@gmail.com>
# Expanded by Serge Sharoff to cover WLD
# This program is free software under GPL 3, see http://www.gnu.org/licenses/

#Our headwords:
# фамилия -0.46701 -0.70868
# ситуация -0.32963 0.053641



from space import Space

import argparse
import collections
import numpy as np
import sys
import smallutils as ut

BATCH_SIZE = 1000


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate embeddings of two languages in a shared space in word translation induction')
    parser.add_argument('-1','--embeddings1', help='the source embeddings')
    parser.add_argument('-2','--embeddings2', help='the target embeddings')
    parser.add_argument('-d', '--dictionary', default=sys.stdin.fileno(), help='the test dictionary file (defaults to stdin)')
    parser.add_argument('--encoding', default='utf-8', action='store_true', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('-l', '--levenshtein', help='the Levenshtein cost dictionary')
    parser.add_argument('-r', '--runname', default='RUN', help='the name for the current run')
    parser.add_argument('-a', '--alpha', default='1', help='the contribution of cos vs Levenshtein')
    parser.add_argument('-k', '--topK', default=10, type=int, help='the number of top candidates to output')
    parser.add_argument('-n', '--neighbors', default=150, type=int, help='the number of neighbors for WLD')
    parser.add_argument('-t', '--threshold', type=int, default=0, help='reduce vocabulary of the model for fast approximate evaluation (0 = off, otherwise typical value is 30,000)')
    parser.add_argument('-v', '--verbosity', type=int, default=1, help='the verbosity level')
    args = parser.parse_args()
    verbosity=args.verbosity
    alpha=float(args.alpha)

    levcosts = ut.readcosts(args.levenshtein)
    # Read input embeddings
    source_sp = Space.build(args.embeddings1,threshold=args.threshold) #,set(wlist)
    source_sp.normalize()
    if verbosity>2:
        print('Read %d source embeddings from %s' % (len(source_sp.row2id), args.embeddings1), file=sys.stderr)

    test_sp = Space.build(args.embeddings2,threshold=args.threshold) #,set(wlist_test)
    test_sp.normalize()
    if verbosity>2:
        print('Read %d target embeddings from %s' % (len(test_sp.row2id), args.embeddings2), file=sys.stderr)
    
    # Read dictionary and compute coverage
    oov = set()
    vocab = set()
    f = open(args.dictionary, encoding=args.encoding, errors='surrogateescape')
    src2trg = collections.defaultdict(set)
    for line in f:
        src, trg = line.rstrip().split('\t')
        try:
            src_ind = source_sp.row2id[src]
            trg_ind = test_sp.row2id[trg]
            src2trg[src_ind].add(trg_ind)
            vocab.add(src)
        except KeyError:
            oov.add(src)
            if verbosity>1:
                print('Out of dict: (%s) vs (%s)' % (src,trg),file=sys.stderr)
    oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
    coverage = len(src2trg) / (len(src2trg) + len(oov))
    if verbosity>1:
        print('Vocab size: %d; Number of pairs: %d' % (len(vocab),len(src2trg)), file=sys.stderr)

    # Compute accuracy
    correctcount = 0
    src, trg = zip(*src2trg.items())
    test_T=test_sp.mat.T
    for i in range(0, len(src2trg), BATCH_SIZE):
        j = min(i + BATCH_SIZE, len(src2trg))
        similarities = source_sp.mat[list(src[i:j])].dot(test_T)
        # nn = np.argmax(similarities, axis=1).tolist()
        nnl=np.argsort(-similarities, axis=1) #scores are negative for reverse sort
        for k in range(j-i):
            id_fs=nnl[k,0:args.neighbors].tolist()  # nn[k]=curlist[0]
            id_fs=id_fs[0]
            scores_f=similarities[k,id_fs]
            if verbosity>4:
                print('Next word %s' % src[i+k], file=sys.stderr)
            w_e= source_sp.id2row[src[i+k]]
            for pos,id_f in enumerate(id_fs):
                w_f=test_sp.id2row[id_f]
                if alpha<1:
                    scores_f[0,pos]= alpha*scores_f[0,pos]+(1-alpha)*(1-ut.iterative_levenshtein(w_e,w_f,levcosts))
            topKlist=np.argsort(-scores_f).tolist()[0][0:args.topK]
            bestid_f=id_fs[topKlist[0]]
            covered=set()
            if bestid_f in trg[i+k]:
                correctcount += 1
            for l in topKlist:
                id_f=id_fs[l]
                if id_f in covered:  # for occasional double vectors
                    continue
                covered.add(id_f)
                if id_f in trg[i+k]:
                    correct = 'Y'
                else:
                    correct = 'N'
                if verbosity>0:
                    goldids=[test_sp.id2row[goldid] for goldid in trg[i+k]]
                    print("%s\t%s\t%s\tR%d\t%.3f\t%s\t{%s}" %(source_sp.id2row[src[i+k]],correct,test_sp.id2row[id_f],l,scores_f[0,l],args.runname,','.join(goldids)))
    print('Coverage:{0:7.2%}  Accuracy total:{1:7.2%}  Accuracy vocab:{2:7.2%}'.format(coverage, correctcount / len(src2trg), correctcount / len(vocab)), file=sys.stderr)


if __name__ == '__main__':
    main()
