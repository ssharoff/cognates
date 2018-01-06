# Detection of cognates

If we consider the task of dictionary induction between related
languages, it is possible to rely on the orthographic similarity
between the two words, for example, *academy* vs *accademia* in
Italian.  This set of tools extends the existing experiments on
cross-lingual word embeddings without using parallel resources by
considering Weighted Levenshtein Distance, where the distance in the
cross-lingual embedding space is weighted by how similar the words are
using weights for the Levenshtein edit operations.  The weights are
also learned on the training dictionary.

The advantage of using weights instead of the standard Levenshtein
Distance is that this decreases the cost of **likely**
substitutions, e.g., the cost of mapping *x* in English to *s* in
Italian is very small (*examined* vs *esaminato*).  Also this works
well across character sets, e.g., mapping *ж* in Russian to
*ż* in Polish (*жизни* vs *życia*).

For a longer description, check the paper:

```
@InProceedings{sharoff2018lgadapt,
  author = {Serge Sharoff},
  title = {Language adaptation experiments via cross-lingual embeddings for related languages},
  booktitle = {Proc LREC},
  year = {2018},
  month = {May},
  address = {Miyazaki, Japan}}
```

This work reuses methods from two experiments on building
cross-lingual models, Dinu, et al, 2014,
[http://clic.cimec.unitn.it/~georgiana.dinu/down/]
and Artetxe, et al, 2016 [https://github.com/artetxem/vecmap]

You start with monolingual vector spaces, the best setup so far involves the vectors from the Facebook group [https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md]

Then, the monolingual vector spaces need to be aligned by either tool.  The cost for the WLD operations are produced by running fast_align ([https://github.com/clab/fast_align]) on the character-separated training dictionary, see also the cost files in the data/ directory.

Finally, for cognate detection run:
```
$ python3 test_tm2.py -a 0.7 -c 2000 -m en-it-300-fasth.trn -1 en-300-fast.dat -2 it-300-fast.dat -l en-it.cost -t 300 -v 1 en-it-test.dic
```

