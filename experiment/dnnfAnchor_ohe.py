#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Explaining d-DNNF (OHE) with Anchor
#   Author: Xuanxiang Huang
#
################################################################################
from __future__ import print_function
import sys
import pandas as pd
from anchor import anchor_tabular
import resource
from math import ceil
from Orange.data import Table
# cat
from decision_tree_ohe import DecisionTree
from ddnnf_ohe import dDNNF
################################################################################


def binarize(dt, samples, preds):
    inst_datatable = Table.from_numpy(dt.datatable.domain, samples, preds)
    inst_datatable_bin = dt.continuizer(inst_datatable)
    return inst_datatable_bin.X
################################################################################


def anchor_call(model, inst, dt, threshold=0.95, verbose=0):
    classifier_fn = lambda x: pd.Series(model.predict_all(list(binarize(dt,x,dt.classifier(x)))))

    timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
            resource.getrusage(resource.RUSAGE_SELF).ru_utime

    explainer = anchor_tabular.AnchorTabularExplainer(
        class_names=list(dt.classes),
        feature_names=dt.features,
        train_data=dt.train_datatable.X)

    exp = explainer.explain_instance(inst,
                                     classifier_fn,
                                     threshold=threshold)
    if verbose:
        print('Anchor: %s' % (' AND '.join(exp.names())))
        print('Precision: %.2f' % exp.precision())
        print('Coverage: %.2f' % exp.coverage())

    timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
            resource.getrusage(resource.RUSAGE_SELF).ru_utime - timer
    if verbose:
        print('  time: {0:.2f}'.format(timer))

    expl_set = set(exp.features())
    print(exp.features())
    print(expl_set)
    print(exp.names())
    assert len(exp.names()) == len(expl_set)

    # length seems incorrect
    return len(expl_set), timer


################################################################################
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == '-bench':
        bench_name = args[1]

        with open(f"{bench_name}", 'r') as fp:
            name_list = fp.readlines()

        for item in name_list:
            # dataset and dt
            data_name = item.strip()
            print(f"############ {data_name} ############")
            data = f"datasets/{data_name}.csv"
            dt_file = f"dt_models/cat/{data_name}.pkl"
            dt = DecisionTree.from_file(dt_file)
            dt_train, dt_test = dt.train()
            # cat
            ddnnf = dDNNF()
            ddnnf.compile(dt, dt.features_bin, dt.features,
                          dt.feat_to_binarize, dt.feat_to_original)

            print(dt.features)
            print(dt.features_bin)

            exps = []
            atimes = []
            d_len = int(len(dt.test_datatable.X) / 2)
            for i in range(d_len):
                print(f"instance {i} of {d_len}:")
                sample = dt.test_datatable.X[i]
                exp, time = anchor_call(ddnnf, sample, dt, verbose=2)
                exps.append(exp)
                atimes.append(time)

            assert len(exps) == d_len
            results = f"{data_name} & {d_len} & {ceil((sum(exps) / d_len) / len(dt.features) * 100):.0f} & "
            results += "{0:.3f} & {1:.3f} & {2:.3f} & {3:.3f}\n".format(sum(atimes), max(atimes), min(atimes), sum(atimes) / len(atimes))
            # with open('results/aaai22_anchor.txt', 'a') as f:
            #     f.write(results)

            print(results)