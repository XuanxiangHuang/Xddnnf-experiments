#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Explaining d-DNNF with Anchor
#   Author: Xuanxiang Huang
#
################################################################################
from __future__ import print_function
import sys
import numpy as np
import pandas as pd
from anchor import anchor_tabular
import resource
from math import ceil
from decision_tree import DecisionTree
from ddnnf import dDNNF
################################################################################


def anchor_call(model, inst, class_names, feature_names, train_data,
                encoder_fn=None, threshold=0.95, verbose=0):
    classifier_fn = lambda x: pd.Series(model.predict_all(list(x)))

    timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
            resource.getrusage(resource.RUSAGE_SELF).ru_utime

    explainer = anchor_tabular.AnchorTabularExplainer(
        class_names=class_names,
        feature_names=feature_names,
        train_data=train_data)

    feat_sample = np.asarray(inst, dtype=np.float32)

    exp = explainer.explain_instance(feat_sample,
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
            dt_file = f"dt_models/bool/{data_name}.pkl"
            dt = DecisionTree.from_file(dt_file)
            dt_train, dt_test = dt.train()
            ddnnf = dDNNF()
            ddnnf.compile(dt)

            exps = []
            atimes = []
            d_len = int(len(dt.X_test) / 2)
            for i in range(d_len):
                print(f"instance {i} of {d_len}:")
                sample = dt.X_test[i]
                exp, time = anchor_call(ddnnf, sample, list(dt.classes), dt.features, dt.X_train, verbose=2)
                exps.append(exp)
                atimes.append(time)

            assert len(exps) == d_len
            results = f"{data_name} & {d_len} & {ceil((sum(exps) / d_len) / len(dt.features) * 100):.0f} & "
            results += "{0:.3f} & {1:.3f} & {2:.3f} & {3:.3f}\n".format(sum(atimes), max(atimes), min(atimes), sum(atimes) / len(atimes))
            # with open('results/aaai22_anchor.txt', 'a') as f:
            #     f.write(results)

            print(results)