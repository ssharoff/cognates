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
Distance is that this can emphasise the **likely** substitutions.
Also this works well across character sets, e.g., *жизни* in Russian
vs *życia* in Polish.

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