grep -v '[0-9]' cs.LOC | shuf -n 1000 | python3 prepare-ner.py LOC v/a/je/na >cs.train-ner
grep -v '[0-9]' hr.LOC | shuf -n 1000 | python3 prepare-ner.py LOC v/i/u/na >hr.train-ner
grep -v '[0-9]' pl.LOC | shuf -n 1000| python3 prepare-ner.py LOC w/i/u/z >pl.train-ner
grep -v '[0-9]' sk.LOC | shuf -n 1000  | python3 prepare-ner.py LOC v/a/na/je >sk.train-ner
grep -v '[0-9]' sl.LOC | shuf -n 1000 | python3 prepare-ner.py LOC v/in/na/je >sl.train-ner
grep -v '[0-9]' cs.ORG |  shuf -n 1000 | python3 prepare-ner.py ORG v/a/je/na >>cs.train-ner
grep -v '[0-9]' cs.PERS | shuf -n 1000 | python3 prepare-ner.py PER v/a/je/na >>cs.train-ner
grep -v '[0-9]' hr.ORG |  shuf -n 1000 | python3 prepare-ner.py ORG v/i/u/na >>hr.train-ner
grep -v '[0-9]' hr.PERS | shuf -n 1000 | python3 prepare-ner.py PER v/i/u/na >>hr.train-ner
grep -v '[0-9]' pl.ORG |  shuf -n 1000 | python3 prepare-ner.py ORG w/i/u/z >>pl.train-ner
grep -v '[0-9]' pl.PERS | shuf -n 1000 | python3 prepare-ner.py PER w/i/u/z >>pl.train-ner
grep -v '[0-9]' sk.ORG |  shuf -n 1000 | python3 prepare-ner.py ORG v/a/na/je >>sk.train-ner
grep -v '[0-9]' sk.PERS | shuf -n 1000 | python3 prepare-ner.py PER v/a/na/je >>sk.train-ner
grep -v '[0-9]' sl.ORG |  shuf -n 1000 | python3 prepare-ner.py ORG v/in/na/je >>sl.train-ner
grep -v '[0-9]' sl.PERS | shuf -n 1000 | python3 prepare-ner.py PER v/in/na/je >>sl.train-ner

grep -v '[a-z0-9]' uk.LOC | egrep -v '[a-z]' | shuf -n 1000 | python3 prepare-ner.py LOC в/і/по/на/має/представляє >uk.train-ner
grep -v '[a-z0-9]' uk.ORG | shuf -n 1000 | python3 prepare-ner.py ORG в/і/по/на/має/представляє >>uk.train-ner
grep -v '[a-z0-9]' uk.PERS | shuf -n 1000 | python3 prepare-ner.py PER в/і/по/на/має/представляє >>uk.train-ner
grep -v '[a-z0-9]' ru.LOC | egrep -v '[a-z]' | shuf -n 1000 | python3 prepare-ner.py LOC в/на/может/является >ru.train-ner
grep -v '[a-z0-9]' ru.ORG | shuf -n 1000 | python3 prepare-ner.py ORG в/на/может/является >>ru.train-ner
grep -v '[a-z0-9]' ru.PERS | shuf -n 1000 | python3 prepare-ner.py PER в/на/может/является >>ru.train-ner
