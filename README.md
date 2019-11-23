# StackelbergEquilibribumSolvers

This package was first made in the winter of 2015 in the state of Tempe at Arizona State University when I was working on a [paper](http://trust.sce.ntu.edu.sg/aamas16/pdfs/p1377.pdf) for AAMAS, 2016.

See `run.sh` in the `src/DOBSS` folders to see how an example `input.txt` can be run.

#### Strategy generation code for web-applications [\[paper\]](http://rakaposhi.eas.asu.edu/aamas16-mtd.pdf):

```bash
cd ./src/DOBSS
python BSG_miqp.py mtd_webapps_input
```

#### Strategy generation code for web-applications that handles switching costs [\[paper\]](http://rakaposhi.eas.asu.edu/AAMAS-2017-MTD.pdf)

```bash
cd ./src/switch_cost_DOBSS
python cost_BSG_miqp.py cost_BSSG_input.txt
```

#### Strategy generation code for IDS placement [\[paper\]](https://yochan-lab.github.io/papers/files/papers/mtd_ids_gamesec.pdf)

```bash
cd ./src/ResourcesHomogeneousScheduleSingleton
python BSG_multi_lp.py BSSG_input.txt
```

The above code provides you with the marginal probabilities. Use the following code to get the mixed strategy distribution (Uses code by Aubrey Clark).
```bash
python strategy_generator.py
```

#### Strategy generation code for deep neural networks [\[paper\]](https://arxiv.org/abs/1705.07213), use the following command:

```
cd ./src/DOBSS
python BSG_miqp.py mtd_neuralnets_input
```
