# Xddnnf-experiments
This repository has code for replicating the experiments in
AAAI'22 paper: Tractable Explanations for d-DNNF Classifiers.

Use the [code](https://github.com/XuanxiangHuang/Xddnnf) here
if you want to try this explainer.

* Experiment results.
* Datasets: [Penn ML Benchmarks](https://epistasislab.github.io/pmlb/) and [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php).
* Decision tree (DT) models.
* d-DNNF/SDD models.

To run scripts, you need:
- [PySAT](https://github.com/pysathq/pysat): for enumerating AXps/CXps.
- [PySDD](https://github.com/wannesm/PySDD): for compiling SDD and find one AXp/CXp of SDD.
- [Orange3](https://github.com/biolab/orange3): for learning DT.
- [Anchor](https://github.com/marcotcr/anchor): for comparison.

## Folders:
#### datasets:
PMLB and UCI Benchmark.

#### models:
decision trees, d-DNNFs, sdds.

#### results:
* [Explaining d-DNNF](results/d-dnnf.csv) and [Explaining SDD](results/sdd.csv):
* [Compilation time](results/compilation_time.csv).


## Files:
'ohe' denotes 'one-hot-encoding'.
* xp*.py: explainer of d-DNNFs/SDDs.

* decision_tree*.py: RODT models (with and without OHE).

* ddnnf.py and ddnnf_ohe.py: d-DNNFs/SDDs manager.

* train.py: training DT/d-DNNF/SDD models (with and without OHE).

* reproduce_experiment.py: explaining d-DNNFs/SDDs via proposed method and via Anchor.