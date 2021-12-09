#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Compiling d-DNNF/SDD Classifiers (one-hot encoding)
#   Author: Xuanxiang Huang
#
################################################################################
# DT
import resource
import sys

import decision_tree_ohe as DTOHE
# SDD/d-DNNF
from ddnnf_ohe import SDD as SDDOHE
from ddnnf_ohe import dDNNF as dDNNFOHE
################################################################################


def compare_pred(mlmodel, model, X_train, X_test):
    # compare prediction of DT and SDD/d-DNNF
    count = 0
    collect = []
    for i in X_train:
        pred1, pred2 = mlmodel.classifier(i), model.predict_one(i)
        if pred1 != pred2:
            print(f'train: {i}, MLmodel: {pred1}, given model: {pred2}')
            collect.append(i)
            count += 1

    for i in X_test:
        pred1, pred2 = mlmodel.classifier(i), model.predict_one(i)
        if pred1 != pred2:
            print(f'test: {i}, MLmodel: {pred1}, given model: {pred2}')
            collect.append(i)
            count += 1
    return count


################################################################################


def compile_sdd_ohe(dt_file, sdd_file, vtree_file):
    # SDD with OHE
    dt = DTOHE.DecisionTree.from_file(dt_file)
    dt_train, dt_test = dt.train()
    # cat
    in_model = dt.dt2bnet()
    Sdd = SDDOHE()

    Sdd.compile(in_model,dt.features_bin,dt.features,
                dt.feat_to_binarize,dt.feat_to_original,
                sdd_file=sdd_file, vtree_file=vtree_file)

    acc_train = round(Sdd.accuracy(dt.train_datatable_bin.X, dt.train_datatable_bin.Y), 3)
    acc_test = round(Sdd.accuracy(dt.test_datatable_bin.X, dt.test_datatable_bin.Y), 3)

    print(f"DT, Train accuracy: {dt_train * 100.0}%")
    print(f"DT, Test accuracy: {dt_test * 100.0}%")

    print(f"SDD (OHE), Train accuracy: {acc_train * 100.0}%")
    print(f"SDD (OHE), Test accuracy: {acc_test * 100.0}%")

    if compare_pred(dt, Sdd, dt.train_datatable_bin.X, dt.test_datatable_bin.X):
        print('DT, SDD are not consistent')
        return None

    return

################################################################################


def rodt_to_ddnnf_ohe(dataset, dt_file, save_dir):
    # d-DNNF with OHE
    file = dataset.split(sep='/')[-1]
    name = file.split(sep='.')[0]
    dt = DTOHE.DecisionTree.from_file(dt_file)
    dt_train, dt_test = dt.train()

    # RODT -> d-DNNF
    ddnnf = dDNNFOHE()
    ddnnf.compile(dt, dt.features_bin, dt.features,
                  dt.feat_to_binarize, dt.feat_to_original)

    acc_train = round(ddnnf.accuracy(dt.train_datatable_bin.X, dt.train_datatable_bin.Y), 3)
    acc_test = round(ddnnf.accuracy(dt.test_datatable_bin.X, dt.test_datatable_bin.Y), 3)

    print(f"DT, Train accuracy: {dt_train * 100.0:.1f}%")
    print(f"DT, Test accuracy: {dt_test * 100.0:.1f}%")

    print(f"d-DNNF (OHE), Train accuracy: {acc_train * 100.0:.1f}%")
    print(f"d-DNNF (OHE), Test accuracy: {acc_test * 100.0:.1f}%")

    if compare_pred(dt, ddnnf, dt.train_datatable_bin.X, dt.test_datatable_bin.X):
        print('DT, d-DNNF are not consistent')
        return None

    print(f'size: {ddnnf.size(ddnnf.root)}')

    if save_dir:
        ddnnf.save_model(ddnnf, filename=f'{save_dir}{name}.pkl')

################################################################################


# -bench xxx -sdd/-ddnnf
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 3 and args[0] == '-bench':
        bench_name = args[1]

        with open(f"{bench_name}", 'r') as fp:
            name_list = fp.readlines()

        for item in name_list:
            # dataset and dt
            data_name = item.strip()
            # dataset and dt
            data_ohe = f"../datasets/{data_name}.csv"
            dt_save_ohe = f"../models/dts/categorical/{data_name}.pkl"
            # save d-DNNF files
            save_ddnnf = f"../models/ddnnfs/"
            # sdd and vtree
            sdd_save_ohe = f"../models/sdds/{data_name}.txt"
            vtree_save_ohe = f"../models/sdds/{data_name}_vtree.txt"
            print(f"############ {data_name} ############")
            if args[2] == '-sdd':
                compile_sdd_time_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                         resource.getrusage(resource.RUSAGE_SELF).ru_utime
                compile_sdd_ohe(dt_save_ohe, None, None)
                compile_sdd_time_end = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                       resource.getrusage(resource.RUSAGE_SELF).ru_utime - compile_sdd_time_start
                print(f"compile {data_name} to SDD in {compile_sdd_time_end:.1f} secs")
                # with open(f'results/compile_time_sdd.txt', 'a') as f:
                #     f.write(f"{data_name} & {compile_sdd_time_end:.1f}\n")
            elif args[2] == '-ddnnf':
                compile_ddnnf_time_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                           resource.getrusage(resource.RUSAGE_SELF).ru_utime
                rodt_to_ddnnf_ohe(data_ohe, dt_save_ohe, None)
                compile_ddnnf_time_end = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                         resource.getrusage(resource.RUSAGE_SELF).ru_utime - compile_ddnnf_time_start
                print(f"compile {data_name} to d-DNNF in {compile_ddnnf_time_end:.1f} secs")
                # with open(f'results/compile_time_ddnnf.txt', 'a') as f:
                #     f.write(f"{data_name} & {compile_ddnnf_time_end:.1f}\n")