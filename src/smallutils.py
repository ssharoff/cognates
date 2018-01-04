#!/usr/bin/env python
# -*- coding: utf-8 -*- 

# import sys
# import codecs

verbosity = 1
# 0 only accuracy
# 1 also errors 
# 2 also top 5
# 3 also correct translations
# 4 also errors in topK
# 5 all with errors
topK = 10
alpha = 0.5
maxexamples = 10

def getword(desc):
    return desc[desc.index('w:')+2:desc.index('~l:')]

def makemaps(descs):
    '''
    makes dictionaries of indices for words and complete descs like
    w:народных~l:народный~m:ADJ-Inan-Gen-Plur~e:ых~c:327 -0.45406 -0.21863
    w:народных~l:народный~m:ADJ-Inan-Loc-Plur~e:ых~c:327 0.12775 -0.20127
    by returning dictionaries with the words and descs to their position in the list
    '''
    desc2ind={}
    word2ind={}
    for i, desc in enumerate(descs):
        desc2ind[desc]=i
        word=getword(desc)
        if word in word2ind:
             word2ind[word].append(i)
        else:
            word2ind[word]=[i]
    return desc2ind, word2ind

def make_y(sp,wlist,annot):
    vocab=[]
    y=[]
    for i,w in enumerate(wlist):
        if w in sp.id2row:
            vocab.append(w)
            y.append(annot[i])
    return vocab,y

def readtrain(filename, vocab_set, multilabel=0):
    '''
    reads training sets in the format
    народных	ADJ-Inan-Gen-Plur
    народных	ADJ-Inan-Loc-Plur
    '''
    X=[]
    y=[]
    # file format: running VBG
    if verbosity>2:
        print("Loading oracle matrix:", filename)

    if multilabel:
        dict={}
    for line in open(filename):
        word, desc = line.strip().split("\t")
        if (not bool(vocab_set)) or (word in vocab_set):  # if the vocab set is known
            if multilabel:
                if word in dict:
                    y[dict[word]].append(desc)
                else:
                    X.append(word)
                    dict[word]=len(X)-1
                    y.append([desc])
            else:
                X.append(word)
                y.append(desc)
    return X, y

def myopen(fn, encoding='utf-8', errors='surrogateescape'):
    if fn.endswith('xz'):
        import lzma
        f= lzma.open(fn,"rt", encoding=encoding, errors=errors)
    elif fn.endswith('gz'):
        import gzip
        f= gzip.open(fn,"rt", encoding=encoding, errors=errors)
    else:
        f= open(fn, encoding=encoding, errors=errors)
    return(f)

def readcosts(fn):
    """
       reading costs from fast_align output
    """
    cost={}
    for l in myopen(fn):
        [s,t,c]=l.split()
        cost[s+t]=1-2.7**float(c)
    return(cost)

def computecost(s,t,cost):
    """
       a simple backoff for the character substitution costs
    """
    if (s==t):
        cost = 0
    elif (s+t) in cost.keys():
        cost = cost[s+t]
    else:
        cost = 1
    return(cost)


def iterative_levenshtein(s, t,cost):
    """ 
    dist[i,j] will contain the Levenshtein distance between the first i characters of s 
    and the first j characters of t
    Modified example from http://www.python-course.eu/levenshtein_distance.php
    """
    rows = len(s)+1
    cols = len(t)+1

    dist = [[0 for x in range(cols)] for x in range(rows)]
    # deletions for source prefixes
    for i in range(1, rows):
        dist[i][0] = i
    # insersions for target prefixes
    for i in range(1, cols):
        dist[0][i] = i
        
    for col in range(1, cols):
        for row in range(1, rows):
            dist[row][col] = min(dist[row-1][col] + computecost('<eps>',t[col-1],cost),  # deletion
                                 dist[row][col-1] + computecost(s[row-1],'<eps>',cost), # insertion
                                 dist[row-1][col-1] + computecost(s[row-1],t[col-1],cost)) # substitution

    return(dist[row][col]/max(cols,rows))
