# Detection of cognates

If we consider the task of dictionary induction between related
languages, it is possible to rely on the orthographic similarity
between the two words.  This set of tools extends the existing
experiments on cross-lingual word embeddings without using parallel
resources by considering Weighted Levenshtein Distance, where the
distance in the cross-lingual embedding space is weighted by how
similar the words are using weights for the Levenshtein edit
operations.  The weights are also learned on the training dictionary.

For a longer description, check the paper:

`
@InProceedings{sharoff18lgadapt,
  author = {Serge Sharoff},
  title = {Language adaptation experiments via cross-lingual embeddings for related languages},
  booktitle = {Proc LREC},
  year = {2018},
  month = {May},
  address = {Miyazaki, Japan}}
`

This work reuses methods from two experiments on building
cross-lingual models, Dinu, et al, 2014,
[http://clic.cimec.unitn.it/~georgiana.dinu/down/]
and Artetxe, et al, 2016 [https://github.com/artetxem/vecmap]