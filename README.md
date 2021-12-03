# Xp d-DNNF/SDD
AAAI'22 paper: Tractable Explanations for d-DNNF Classifiers.

* Experiment results.
* Datasets: [Penn ML Benchmarks](https://epistasislab.github.io/pmlb/) and [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php).
* Decision tree (DT) models.
* d-DNNF/SDD models.

To run scripts, you need:
- [PySAT](https://github.com/pysathq/pysat): for enumerating AXps/CXps.
- [PySDD](https://github.com/wannesm/PySDD): for compiling SDD and find one AXp/CXp of SDD.
- [Orange3](https://github.com/biolab/orange3): for learning DT.

## Folders:
#### datasets:
PMLB and UCI Benchmark.

#### dt_models:
Trained DT models, 'bool' indicates all features are binary,
'cat' indicates there are categorical features.
(Each dataset was randomly split into training data (80%) and testing data (20%).
The test instances reported in the paper are the first half of the testing data.)

#### sdd_models:
sdd files and vtree files, you can check SDD size here.

#### ddnnf_models:
d-DNNF models.

#### results:
* aaai22_*: all results presented in the paper.
   (datasets are randomly split into 80% for training and 20% for testing,
   explained instances are the first half of the testing data).
* compile_*: compilation time.
* dt_info: accuracy and size of DT.
* ddnnf_* and sdd_*: explaining all instances of BOOL datasets and training instances of CAT datasets.


## Files:
'ohe' denotes 'one-hot-encoding'.
* train_dt.py: training DT models (with and without OHE).

* compile_ddnnf_sdd*.py: compiling RODTs into d-DNNFs/SDDs.

* check_rodt.py: check if DT is read-once.

* xp*.py: explainer of d-DNNFs/SDDs.

* explain_*.py: running experiments.

* decision_tree*.py: RODT models (with and without OHE).

* dnnfAnchor*.py: explaining ddnnf with Anchor.

* ddnnf.py and ddnnf_ohe.py: d-DNNFs/SDDs manager.