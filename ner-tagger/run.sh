cat cs.LOC | shuf -n 1000 | python3 prepare-ner.py LOC v/a/je/na >cs.train-ner
shuf -n 1000 cs.ORG | python3 prepare-ner.py ORG v/a/je/na >>cs.train-ner
shuf -n 1000 cs.PER | python3 prepare-ner.py PER v/a/je/na >>cs.train-ner
cat hr.LOC | shuf -n 1000 | python3 prepare-ner.py LOC v/i/u/na >hr.train-ner
shuf -n 1000 hr.ORG | python3 prepare-ner.py ORG v/i/u/na >>hr.train-ner
shuf -n 1000 hr.PER | python3 prepare-ner.py PER v/i/u/na >>hr.train-ner
cat pl.LOC | shuf -n 1000| python3 prepare-ner.py LOC w/i/u/z >pl.train-ner
shuf -n 1000 pl.ORG | python3 prepare-ner.py ORG w/i/u/z >>pl.train-ner
shuf -n 1000 pl.PER | python3 prepare-ner.py PER w/i/u/z >>pl.train-ner
cat ru.LOC |shuf -n 1000 | python3 prepare-ner.py LOC в/и/о/на >ru.train-ner
shuf -n 1000 ru.ORG | python3 prepare-ner.py ORG в/и/о/на >>ru.train-ner
shuf -n 1000 ru.PER | python3 prepare-ner.py PER в/и/о/на >>ru.train-ner
cat sk.LOC | shuf -n 1000  | python3 prepare-ner.py LOC v/a/na/je >sk.train-ner
shuf -n 1000 sk.ORG | python3 prepare-ner.py ORG v/a/na/je >>sk.train-ner
shuf -n 1000 sk.PER | python3 prepare-ner.py PER v/a/na/je >>sk.train-ner
cat sl.LOC | shuf -n 1000 | python3 prepare-ner.py LOC v/in/na/je >sl.train-ner
shuf -n 1000 sl.ORG | python3 prepare-ner.py ORG v/in/na/je >>sl.train-ner
shuf -n 1000 sl.PER | python3 prepare-ner.py PER v/in/na/je >>sl.train-ner
cat uk.LOC | shuf -n 1000 | python3 prepare-ner.py LOC в/і/по/на >uk.train-ner
egrep -v '[a-z]' uk.ORG | shuf -n 1000 | python3 prepare-ner.py ORG в/і/по/на >>uk.train-ner
egrep -v '[a-z]' uk.PER | shuf -n 1000 | python3 prepare-ner.py PER в/і/по/на >>uk.train-ner
