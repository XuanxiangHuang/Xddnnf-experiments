# Xddnnf-experiments
This repository contains sources for replicating the experiments performed in 
our AAAI'22 paper: Tractable Explanations for d-DNNF Classifiers.

To access the experimental results:
- [Explaining d-DNNF](results/d-dnnf.csv) and [Explaining SDD](results/sdd.csv).
- [Compilation time](results/compilation_time.csv).

Use the [code](https://github.com/XuanxiangHuang/Xddnnf) here
if you want to try this explainer.

## Getting Started
To run the scripts, you need to install the following Python packages:
- [PySAT](https://github.com/pysathq/pysat): for enumerating AXps/CXps.
- [PySDD](https://github.com/wannesm/PySDD): for compiling SDD and find one AXp/CXp of SDD.
- [Orange3](https://github.com/biolab/orange3): for learning DT.
- [Anchor](https://github.com/marcotcr/anchor): for computing Anchor explanations and compare with AXp's.

## Examples

To train a decision tree:
```
python3 train.py -d 6 -l 0.8:0.8 -c dt -D datasets/corral.csv -s tmp/dt_corral
```

To compile an sdd from a decision tree:
```
python3 train.py -c sdd -D datasets/adult.csv -f models/dts/categorical/adult.pkl -o -s tmp/adult
```

To reproduce experiments:
```
python3 experiment.py -bench data_cat_list.txt -sdd -ohe
```
