#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

# Copyright (C) 2017  Serge Sharoff
# The tool converts gazetteers into training files with appropriate start and end words
# 1 type (per, loc, org, misc)
# 2 start/end words separated by /
# и Sl I-S O O
# Быстрый Npmsn I-N B-loc
# Берег Npmsn I-N I-loc
# в Sl I-S O O

import sys, re
import random

outformat="%s Npmsn I-N %s"
outbreak="%s Sl I-S O %s"

punct=r'([\[\]\(\),.-/":<>”?“!»«+\'|_’;%‘*=&°@ ~>§©])'

def tokeniseall(s):
    return re.sub(punct,r' \1 ',s)

gtype=sys.argv[1].lower()
endwords=sys.argv[2].split('/')

for l in sys.stdin:
    wlist=tokeniseall(l).split()
    print(outbreak % (endwords[random.randrange(len(endwords))], 'O'))
    print(outformat % (wlist[0], 'B-'+gtype))
    for i in range(1,len(wlist)):
        print(outformat % (wlist[i], 'I-'+gtype))
    print(outbreak % (endwords[random.randrange(len(endwords))], 'O'))
    print("")
