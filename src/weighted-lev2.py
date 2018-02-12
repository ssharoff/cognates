#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import sys
import smallutils
import io

cost=smallutils.readcosts(sys.argv[1])
if len(sys.argv)>2:
    cost=cost.update(smallutils.readcosts(sys.argv[2]))

#print(iterative_levenshtein(u'англійські',u'английские',cost))
#print(iterative_levenshtein(u'англійські',u'английски',cost))
#print(iterative_levenshtein(u'англійські',u'английском',cost))

for line in sys.stdin:
    f=line.rstrip().split()
    if len(f)>1:
        l1=smallutils.iterative_levenshtein(f[0],f[1],cost)
        print('%s\t%s\t%.3f' % (f[0],f[1],l1))

