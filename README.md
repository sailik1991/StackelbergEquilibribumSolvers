# StackelbergEquilibribumSolvers

This package was first made in the winter of 2015 in the state of Tempe at Arizona State University when I was working on a [paper](http://trust.sce.ntu.edu.sg/aamas16/pdfs/p1377.pdf) for AAMAS, 2016. I have later added opensource implementation of the files using Google's [ORTools](https://developers.google.com/optimization) (these files are also unit-tested).

If you plan to use the ORTools too (insted of gurobi), here is the venv setup
```bash
# From the repository root, create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install OR-Tools
pip install ortools

# To deactivate when done
deactivate
```

#### Strategy generation code for web-applications [\[paper\]](http://rakaposhi.eas.asu.edu/aamas16-mtd.pdf):

```bash
cd ./src/DOBSS
python BSG_miqp.py mtd_webapps_input
```

Running unit-tests (supported for the *_ortools.py version as it doesn't need a gurobi license.)
```bash
# From the repository root, activate the virtual environment
source .venv/bin/activate

# Run all DOBSS tests
python3 -m unittest discover src/DOBSS/tests/ -v

# Run a specific test file
python3 src/DOBSS/tests/test_bsg_miqp_ortools.py
python3 src/DOBSS/tests/test_whatToFix_ortools.py

```

#### Strategy generation code for web-applications that handles switching costs [\[paper\]](http://rakaposhi.eas.asu.edu/AAMAS-2017-MTD.pdf)

```bash
cd ./src/switch_cost_DOBSS
python cost_BSG_miqp.py cost_BSSG_input.txt
```

Running unit-tests
```bash
# From the repository root, activate the virtual environment
source .venv/bin/activate

# Run all DOBSS tests
python3 -m unittest discover src/switch_cost_DOBSS/tests/ -v

# Run a specific test file
python3 src/switch_cost_DOBSS/tests/test_cost_bsg_miqp_ortools.py
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

```bash
cd ./src/DOBSS
python BSG_miqp.py mtd_neuralnets_input
```