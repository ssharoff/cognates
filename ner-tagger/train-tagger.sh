#!/bin/bash
THEANO_FLAGS=device=cuda python3 train.py --train dataset/$1.train --dev dataset/$1.testa --test dataset/$1.testb -o $1.wl -s iob -l 1 -z 1 -D 0.5 -a 1 -p $2 -w $3 $4

