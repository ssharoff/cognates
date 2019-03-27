#!/bin/bash
#$ -cwd -V
#$ -l h_rt=48:00:00
#$ -m e
#$ -l h_vmem=4G

module add mkl
module add python
module add python-libs

DATA='data'
OUTPUT='out'

L1=$1  # en
L2=$2  # it
SUF=$3 # 300-fasttext.dat

SRC_EMB=$L1-$SUF
TRG_EMB=$L2-$SUF
SRCTRG_EMB=$L2$L1-$SUF

TRAIN_DIC=$L1-$L2-train.dic
TEST_DIC=$L1-$L2-test.dic

mkdir -p $OUTPUT/

if [ ! -f $OUTPUT/$SRC_EMB ]; then
  echo "normalize_embeddings.py unit center -i $DATA/$SRC_EMB -o $OUTPUT/unit-center/$SRC_EMB"
  time python3 src/normalize_embeddings.py unit center -i $DATA/$SRC_EMB -o $OUTPUT/$SRC_EMB
fi
if [ ! -f $OUTPUT/$TRG_EMB ]; then
  echo "normalize_embeddings.py unit center -i $TRG_EMB -o $OUTPUT/unit-center/$TRG_EMB"
  time python3 src/normalize_embeddings.py unit center -i $DATA/$TRG_EMB -o $OUTPUT/$TRG_EMB
fi

echo "building mapping for $SRC_EMB using $TRAIN_DIC"
if [ ! -f $OUTPUT/$SRCTRG_EMB ]; then
  time python3 src/project_embeddings.py --orthogonal $OUTPUT/$SRC_EMB $OUTPUT/$TRG_EMB -d $DATA/$TRAIN_DIC -o $OUTPUT/$SRCTRG_EMB
fi

python3 src/eval_translation1.py -1 $OUTPUT/$SRCTRG_EMB -2 $OUTPUT/$TRG_EMB -d $DATA/$TEST_DIC >$L1-$L2.out
