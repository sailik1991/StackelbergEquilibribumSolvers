# StackelbergEquilibribumSolvers

This package was first made in the winter of 2015 in the state of Tempe at Arizona State University when I was working on a [paper](http://trust.sce.ntu.edu.sg/aamas16/pdfs/p1377.pdf) for AAMAS, 2016.

See `run.sh` in the `src/DOBSS` folders to see how an example `input.txt` can be run.

+ To run the strategy generation code for web-applications [\[paper\]](http://rakaposhi.eas.asu.edu/AAMAS-2017-MTD.pdf), use the following command:

```
gurobi.sh BSG_miqp.py mtd_webapps_input
```

+ To run the strategy generation code for deep neural networks [\[paper\]](https://arxiv.org/abs/1705.07213), use the following command:

```
gurobi.sh BSG_miqp.py mtd_neuralnets_input
```
