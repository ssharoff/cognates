#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

# Copyright (C) 2016-2017  Serge Sharoff
# This program is free software under GPL 3, see http://www.gnu.org/licenses/
'''
The script predicts properties of embeddings. Typical examples:
predict_morph1.py -1 $VECTOR -t $POS.train -k 10 -m 1 -c MLP -f 1 -v 4

predict_morph1.py -1 $VECTOR1 -2 $VECTOR2 -t $POS1.train -d $POS2.train -m 1 -c MLP -f 1 -v 4
(this takes a better resourced L1 and tests on less resourced L2
'''


import argparse
import numpy as np
import sys

import smallutils as ut
from space import Space

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, label_ranking_average_precision_score, f1_score
from sklearn import metrics
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import ParameterGrid

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Project the source embeddings into the target embedding space maximizing the squared Euclidean distances for the given dictionary')
    parser.add_argument('-1','--embeddings1', help='the source embeddings')
    parser.add_argument('-2','--embeddings2', help='the target embeddings')
    parser.add_argument('-t', '--train', help='the training file')
    parser.add_argument('-d', '--dev', help='the development file')
    parser.add_argument('-k', '--kfolds', type=int, default=5, help='K for cross-validation')
    parser.add_argument('-m', '--multilabel', type=int, default=0, help='use multilabel')
    parser.add_argument('-p', '--params', type=int, default=0, help='for HP tuning')
    parser.add_argument('-c', '--classifier', help='the classifier: LR, MLP, SVM or kNN')
    parser.add_argument('-s', '--solver', default='adam', help='the appropriate solver: sgd, adam, sag, liblinear, etc')
    parser.add_argument('-f', '--fallback', type=bool, default=False, help='use kNN as fallback')
    parser.add_argument('--encoding', default='utf-8', action='store_true', help='the character encoding for input/output (defaults to utf-8)')
    # parser.add_argument('-l', '--levenshtein', help='the Levenshtein cost dictionary')
    # parser.add_argument('-a', '--alpha', default='0.7', help='the contribution of cos vs Levenshtein')
    parser.add_argument('-v', '--verbosity', type=int, default=1, help='the verbosity level')
    args = parser.parse_args()
    verbosity=args.verbosity
    ut.verbosity=args.verbosity

    trainfile = ut.myopen(args.train, encoding=args.encoding, errors='surrogateescape')
    wlist, annot = ut.readtrain(args.train,{},args.multilabel)
    if verbosity>2:
        print('Read %d train examples from %s' % (len(annot), args.train))

    if (args.dev):
        testfile = ut.myopen(args.dev, encoding=args.encoding, errors='surrogateescape')
        wlist_test, annot_test = ut.readtrain(args.dev,{},args.multilabel)
        if verbosity>2:
            print('Read %d test examples from %s' % (len(annot_test), args.dev))
        
    source_sp = Space.build(args.embeddings1) #,set(wlist)
    if verbosity>2:
        print('Read %d source embeddings from %s' % (len(source_sp.row2id), args.embeddings1))

    if (args.dev):
        if args.embeddings2:
            test_sp = Space.build(args.embeddings2) #,set(wlist_test)
            if verbosity>2:
                print('Read %d target embeddings from %s' % (len(test_sp.row2id), args.embeddings2))
        else:
            test_sp=source_sp
    
    # Build word to index maps
    # src_desc2ind, src_word2ind = ut.makemaps(src_descs)
    
    vocab,y=ut.make_y(source_sp.id2row,wlist,annot)
    if verbosity>2:
        print('Total training lexicon of %d examples' % len(y))
    if (args.dev):
        vocab_test,y_test=ut.make_y(test_sp.id2row,wlist_test,annot_test)
        if verbosity>2:
            print('Total testing lexicon of %d examples' % len(y_test))

        
    # # Read dictionary
    # f = ut.myopen(args.dictionary, encoding=args.encoding, errors='surrogateescape')
    # src_indices = []
    # trg_indices = []
    # for line in f:
    #     elts = line.split()  # in case there are extra fields
    #     src, trg = elts[0:2]
    #     try:
    #         src_ind = src_word2ind[src]
    #         trg_ind = trg_word2ind[trg]
    #         for i in src_ind:
    #             for j in trg_ind:
    #                 src_indices.append(i)
    #                 trg_indices.append(j)
    #     except KeyError:
    #         if verbosity>1:
    #             print('WARNING: OOV dictionary entry ({0} - {1})'.format(src, trg), file=sys.stderr)
    # if verbosity>2:
    #     print('Read %d translations from %s' % (len(src_indices), args.dictionary), file=sys.stderr)

    # X_train, X_test, y_train, y_test = train_test_split(vocab, y, test_size=args.subset)
    # if verbosity>2:
    #     print('Train lexicon: %d, test lexicon: %d' % (len(y_train),len(y_test)))
    # X_train_mat = source_sp.mat[[source_sp.row2id[el] for el in X_train],:]
    X_train_mat = source_sp.mat[[source_sp.row2id[el] for el in vocab],:]
    if args.dev:
        X_test_mat = test_sp.mat[[test_sp.row2id[el] for el in vocab_test],:]
    if args.multilabel:
        mlb = MultiLabelBinarizer()
        if (args.dev):
            yfull = y + y_test
            yfull = mlb.fit_transform(yfull)
            y = yfull[0:len(y)]
            y_test = yfull[len(y):]
        else:
            y = mlb.fit_transform(y)
    
    if args.classifier == 'LR':
        clf = LogisticRegression(solver=args.solver, max_iter=100, random_state=42,
                                 multi_class='multinomial')
    elif args.classifier == 'MLP':

        if (args.params>0):
            p=ParameterGrid({'hl': [(50,), (75,), (75,75,75), (75,50,25)],
                             'mi':[50,100,200],
                             'alpha':[1e-3,1e-4,1e-5],
                             'lri':[.1,1e-2,1e-3],
                             'act':['relu', 'logistic', 'tanh' ]})[args.params-1]
        else:
            p={'mi': 50, 'lri': 0.1, 'hl': (150,), 'alpha': 0.001, 'act': 'tanh'} # 'hl':(200,) is better
        print(p)
        clf = MLPClassifier(hidden_layer_sizes=p['hl'], max_iter=p['mi'], alpha=p['alpha'],
                    solver=args.solver, verbose=0, tol=1e-4, random_state=1,
                    learning_rate_init=p['lri'], activation=p['act'], early_stopping = True)
    elif args.classifier == 'SVC':
        clf = SVC()
    elif args.classifier == 'kNN':
        clf=KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto')
    elif args.classifier == 'RF':
        clf=RandomForestClassifier()
    if verbosity>2:
        print('Classifier: %s' % args.classifier)
    if args.fallback:
        clf_fallback=KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='auto')

    # clf.fit(X_train_mat, y_train)
    # y_pred = clf.predict(X_test_mat)
    # print(confusion_matrix(y_test,y_pred))
    # print(classification_report(y_test,y_pred))
    # print(accuracy_score(y_test,y_pred))  # clf.predict_proba(X_test_mat)
    if (args.dev):
        clf.fit(X_train_mat, y)
        predictions = clf.predict(X_test_mat)
        if args.multilabel:
            if args.fallback:
                clf_fallback.fit(X_train_mat, y)
                pred_fallback = mlb.inverse_transform(clf_fallback.predict(X_test_mat))
            print("Label ranking average precision: %0.3f" % metrics.label_ranking_average_precision_score(y_test,predictions))
            print("F1 score: %0.3f" % metrics.f1_score(y_test,predictions,average='samples'))
            i=0
            if verbosity>3:
                for gold,pred in zip(mlb.inverse_transform(y_test),mlb.inverse_transform(predictions)):
                    if args.fallback and len(pred)==0: # occasionaly no output is given
                        pred=pred_fallback[i]
                    print("%0.1f\t%s\t%s\t%s" % (len(set(gold).intersection(set(pred)))/len(gold),vocab_test[i],gold,pred))
                    i+=1
        else:
            print("F1 Macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    else:
        predictions = cross_val_predict(clf, X_train_mat, y, n_jobs=-2, cv=args.kfolds)
        scores = cross_val_score(clf, X_train_mat, y, n_jobs=-2, cv=args.kfolds, scoring='f1_macro') 
        print("F1 Macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        if args.multilabel:
            if args.fallback:
                pred_fallback = mlb.inverse_transform(cross_val_predict(clf_fallback, X_train_mat, y, n_jobs=-2, cv=args.kfolds))
            i=0
            if verbosity>3:
                for gold,pred in zip(mlb.inverse_transform(y),mlb.inverse_transform(predictions)):
                    if args.fallback and len(pred)==0: # occasionaly no output is given
                        pred=pred_fallback[i]
                        print("%1d\tFor %s (%s) fallback: %s" % (gold==pred,vocab[i],gold,pred), file=sys.stderr)
                    print("%0.1f\t%s\t%s\t%s" % (len(set(gold).intersection(set(pred)))/len(gold),vocab[i],gold,pred))
                    i+=1
    
if __name__ == '__main__':
    main()
