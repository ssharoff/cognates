# Cross-lingual embeddings for related languages 

An embedding space is a vector space in which words with similar
meanings are located close to each other.  Since 2011 there have been
a number of studies devoted to constructing cross-lingual embedding
spaces, see an [overview](https://github.com/artetxem/vecmap).

If we consider the task of dictionary induction between related
languages, it is possible to rely on the orthographic similarity
between the two words, for example, *academy* in English vs *accademia* in
Italian.  This set of tools extends the existing experiments on
cross-lingual word embeddings without using parallel resources by
considering Weighted Levenshtein Distance, where the distance in the
cross-lingual embedding space is weighted by how similar the words are
using weights for the Levenshtein edit operations.  The weights are
also learned from the training dictionary.

The advantage of using weights instead of the standard Levenshtein
Distance is that this decreases the cost of **likely** substitutions,
e.g., the cost of mapping *x* in English to *s* in Italian is very
small (*examined* vs *esaminato*).  Also this works well across
character sets, e.g., mapping *ж* in Russian to *ż* in Polish (*жизни*
vs *życia*).  The cost for the WLD operations are produced by running
[fast_align](https://github.com/clab/fast_align) on the
character-separated training dictionary, see also the cost files in
the data/ directory.

For a longer description, check the paper:

```
@Article{sharoff19jnle,
  author = 	 {Serge Sharoff},
  title = 	 {Finding next of kin: Cross-lingual embedding spaces for related languages},
  journal = 	 {Journal of Natural Language Engineering},
  year = 	 2019,
  volume = 	 25}
```
 [http://corpus.leeds.ac.uk/serge/publications/2019-ftd.pdf]

This work reuses methods from two experiments on building
cross-lingual models, Dinu, et al, 2014,
[http://clic.cimec.unitn.it/~georgiana.dinu/down/]
and Artetxe, et al, 2016 [https://github.com/artetxem/vecmap]

You start with monolingual vector spaces, the best setup so far involves the FastText vectors from the Facebook group [https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md]

Then, the monolingual vector spaces need to be aligned using the orthogonal transform, e.g.
```
./align-spaces.sh en it 300-fasttext.dat
```
The output is in the format of [trec_eval](http://trec.nist.gov/trec_eval/), which makes it easy to run various evaluation metrics (note that success@1trec_eval  corresponds to precision@1 in the lexicon building community).

For cognate detection you need a word list for the source language.  As a simple approximation a word list can be taken from the existing vectors:
```
cut -f 1 -d ' ' out/en-300-fasttext.vec | grep '^[a-z-]*$' | perl -pe 'undef $_ if length($_)<=4' | sed 's/$/\tnone/' >en.wl
```
Then run congnate detection for the aligned vectors:
```
python3 src/eval_translation1.py -a 0.73 -l data/en-it.cost -1 out/iten-300-fasttext.vec -2 out/it-300-fasttext.vec -d en.wl | cut -f 1,3,5 | sort -nsrk3,3 >en-it.trans
```

This produces a list of *possible* cognates: words at the top of the list are likely to be cognates, words at the bottom are either incorrect translations, or correct translation which are not cognates.  You need to estimate the threshold for the most likely cognates.  If you have a small dictionary of gold-standard cognates, it might be useful to estimate the positions of those words in the full list:

```
grep -nFf en-it-test.dic en-it.trans
```
In my experience the cognates have the similarity score above 0.5, but the exact useful value depends on the language pair and the quality constraints.

Once you have a list reliable cognates in `en-it.cognates', they can be used in the second round for building embedding spaces:
```
time python3 src/project_embeddings.py --orthogonal out/iten-300-fasttext.vec out/it-300-fasttext.vec -d en-it.cognates -o out/it2en-300-fasttext.vec
```

# Shared Panslavonic embedding space for Language Adaptation

Cognate detection is useful in the context of Language Adaptation,
when an embedding space is shared across related languages and can be
used to improve NLP tasks in a lesser resourced language.

This is the list of embeddings for the shared Panslavonic space:

* [Czech](http://corpus.leeds.ac.uk/serge/cognates/cs-300-panslav.vec.xz)
* [English](http://corpus.leeds.ac.uk/serge/cognates/en-300-panslav.vec.xz)
* [Croatian](http://corpus.leeds.ac.uk/serge/cognates/hr-300-panslav.vec.xz)
* [Polish](http://corpus.leeds.ac.uk/serge/cognates/pl-300-panslav.vec.xz)
* [Russian](http://corpus.leeds.ac.uk/serge/cognates/ru-300-panslav.vec.xz)
* [Slovak](http://corpus.leeds.ac.uk/serge/cognates/sk-300-panslav.vec.xz)
* [Slovene](http://corpus.leeds.ac.uk/serge/cognates/sl-300-panslav.vec.xz)
* [Ukrainian](http://corpus.leeds.ac.uk/serge/cognates/uk-300-panslav.vec.xz)

The lists/ directory in this github repository contains various dictionaries of cognates built from the shared Panslavonic space (and beyond, including Germanic, Romance and Uralic languages).  I keep the language pairs I experiment with.  If you want a  specific language pair, I can probably produce it too.  What this needs is just a corpus (of more than 50 million words) and a seed dictionary.

As an example, a Named-Entity Recognition (NER) tagger covering Croatian, Czech, Polish, Russian, Slovak, Slovene and Ukrainian has been built using the shared space.  It's simply an adaptation of a monolingual [NER tagger](https://github.com/glample/tagger)

The NER tagger requires Theano and it can be run as:
```
./run-tagger.sh sl-cs input_file
```
for running the Czech model transferred from Slovenian.
