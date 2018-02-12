#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

# Copyright (C) 2017  Serge Sharoff
# The tool gives the coverage of each line per dictionary.

import sys, re

punct=r'([\[\]\(\),.-/":<>”?“!»«+\'|_’;%‘*=&°@ ~>§©])'
# also 

def tokeniseall(s):
    return re.sub(punct,r' \1 ',s)

vocab = set()

for l in open(sys.argv[1]):
    vocab.add(l.strip())

for l in sys.stdin:
    wend=l.find('\t') # ignore annotations
    words=tokeniseall(l[0:wend].lower()).split()
    wcount=0
    for w in words:
        if w in vocab:
            wcount+=1
    print("%.3f" % float(wcount/len(words)) )
