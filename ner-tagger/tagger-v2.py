#!/usr/bin/env python3
# Serge Sharoff 2017, University of Leeds,
# a modified version of the NER tagger from: https://github.com/glample/tagger
# For the full context see: https://github.com/ssharoff/panslavonic

import os
import time
import codecs
import optparse
import numpy as np
from loader import prepare_sentence
from utils import create_input, iobes_iob, zero_digits, tokeniseall
from model import Model

optparser = optparse.OptionParser()
optparser.add_option(    "-m", "--model", default="",    help="Model location")
optparser.add_option(    "-i", "--input", default="",    help="Input file or directory location")
optparser.add_option(    "-o", "--output", default="",    help="Output file or directory  location")
optparser.add_option(    "-d", "--delimiter", default="__",    help="Delimiter to separate words from their tags")
opts = optparser.parse_args()[0]

# Check parameters validity
assert opts.delimiter
assert os.path.isdir(opts.model), "Directory %s does not exist" % opts.model
if os.path.isdir(opts.input):
    flist=[opts.input+'/'+fn for fn in os.listdir(opts.input)]
elif os.path.isfile(opts.input):
    flist=[opts.input]
assert os.path.isfile(flist[0]), "Input texts %s do not exist" % flist[0]

if os.path.isdir(opts.output):
    outflist=[opts.output+'/'+fn for fn in os.listdir(opts.input)]
else:
    outflist=[opts.output]

assert len(outflist) == len(flist)

# Load existing model
print("Loading model...")
model = Model(model_path=opts.model)
parameters = model.parameters

# Load reverse mappings
word_to_id, char_to_id, tag_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
]

# Load the model
_, f_eval = model.build(training=False, **parameters)
model.reload()

start = time.time()

count = 0
print('Tagging...')
for i in range(len(flist)):
    fn=flist[i]
    f_output = codecs.open(outflist[i], 'w', 'utf-8')
    with codecs.open(fn, 'r', 'utf-8') as f_input:
        for line in f_input:
            line = tokeniseall(line.rstrip())
            words = line.split()
            # print(line)
            if line and len(words)>1:
                # Lowercase sentence
                if parameters['lower']:
                    line = line.lower()
                    # Replace all digits with zeros
                if parameters['zeros']:
                    line = zero_digits(line)
                    # Prepare input
                sentence = prepare_sentence(words, word_to_id, char_to_id,
                                            lower=parameters['lower'])
                input = create_input(sentence, parameters, False)
                # Decoding
                if parameters['crf']:
                    y_preds = np.array(f_eval(*input))[1:-1]
                else:
                    y_preds = f_eval(*input).argmax(axis=1)
                y_preds = [model.id_to_tag[y_pred] for y_pred in y_preds]
                    # Output tags in the IOB2 format
                if parameters['tag_scheme'] == 'iobes':
                    y_preds = iobes_iob(y_preds)
                    # Write tags
                assert len(y_preds) == len(words)
                f_output.write('%s\n' % ' '.join('%s%s%s' % (w, opts.delimiter, y)
                                                 for w, y in zip(words, y_preds)))
            else:
                f_output.write('\n')
                count += 1
            if count % 100 == 0:
                print('Processed %d words in %s' % (count,fn))

print('---- %i lines tagged in %.4fs ----' % (count, time.time() - start))
f_output.close()
