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

## Compilation Time (in seconds):
#### Compiling RODTs to d-DNNFs/SDDs.
| Dataset               | d-DNNF | SDD |
|-----------------------|--------|-----|
| corral                | 0.0    | 0.0 |
| mofn_3_7_10           | 0.5    | 0.1 |
| mux6                  | 0.1    | 0.0 |
| parity5+5             | 2.0    | 0.3 |
| spect                 | 0.2    | 0.1 |
| threeOf9              | 0.2    | 0.1 |
| adult                 | 11.2   | 2.7 |
| chess                 | 0.8    | 0.4 |
| compas                | 10.2   | 1.2 |
| german                | 5.3    | 3.5 |
| kr_vs_kp              | 0.9    | 0.4 |
| lending               | 7.9    | 1.3 |
| Mammographic-mass     | 0.5    | 0.1 |
| monk1                 | 0.3    | 0.1 |
| monk2                 | 0.2    | 0.1 |
| monk3                 | 0.1    | 0.1 |
| postoperative-patient | 0.1    | 0.1 |
| primary-tumor         | 0.2    | 0.1 |
| promoters             | 0.2    | 0.2 |
| recidivism            | 22.6   | 5.9 |
| tic_tac_toe           | 0.8    | 0.2 |
| vote                  | 0.1    | 0.1 |





