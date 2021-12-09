#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Compiling d-DNNF/SDD Classifiers
#
################################################################################
# DT
import resource
import sys
import decision_tree as DT
# SDD/d-DNNF
from ddnnf import SDD, dDNNF
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


def compile_sdd(dt_file, sdd_file, vtree_file):
    # SDD
    dt = DT.DecisionTree.from_file(dt_file)
    dt_train, dt_test = dt.train()
    in_model = dt.bdt2bnet()
    Sdd = SDD()
    Sdd.compile(in_model, dt.features, sdd_file=sdd_file, vtree_file=vtree_file)

    acc_train = round(Sdd.accuracy(dt.X_train, dt.y_train), 3)
    acc_test = round(Sdd.accuracy(dt.X_test, dt.y_test), 3)

    print(f"DT, Train accuracy: {dt_train * 100.0}%")
    print(f"DT, Test accuracy: {dt_test * 100.0}%")

    print(f"SDD, Train accuracy: {acc_train * 100.0}%")
    print(f"SDD, Test accuracy: {acc_test * 100.0}%")

    if compare_pred(dt, Sdd, dt.X_train, dt.X_test):
        print('DT, SDD are not consistent')
        return None

    return


def rodt_to_ddnnf(dataset, dt_file, save_dir):
    # d-DNNF
    file = dataset.split(sep='/')[-1]
    name = file.split(sep='.')[0]
    dt = DT.DecisionTree.from_file(dt_file)
    dt_train, dt_test = dt.train()

    # RODT -> d-DNNF
    ddnnf = dDNNF()
    ddnnf.compile(dt)

    acc_train = round(ddnnf.accuracy(dt.X_train, dt.y_train), 3)
    acc_test = round(ddnnf.accuracy(dt.X_test, dt.y_test), 3)

    print(f"DT, Train accuracy: {dt_train * 100.0}%")
    print(f"DT, Test accuracy: {dt_test * 100.0}%")

    print(f"d-DNNF, Train accuracy: {acc_train * 100.0}%")
    print(f"d-DNNF, Test accuracy: {acc_test * 100.0}%")

    if compare_pred(dt, ddnnf, dt.X_train, dt.X_test):
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
            data_name = item.strip()
            # dataset and dt
            data = f"datasets/{data_name}.csv"
            dt_save = f"dt_models/bool/{data_name}.pkl"
            # save d-DNNF files
            save_ddnnf = f"ddnnf_models/"
            # sdd and vtree
            sdd_save = f"sdd_models/{data_name}.txt"
            vtree_save = f"sdd_models/{data_name}_vtree.txt"
            print(f"############ {data_name} ############")
            if args[2] == '-sdd':
                compile_sdd_time_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                         resource.getrusage(resource.RUSAGE_SELF).ru_utime
                compile_sdd(dt_save, None, None)
                compile_sdd_time_end = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                       resource.getrusage(resource.RUSAGE_SELF).ru_utime - compile_sdd_time_start
                print(f"compile {data_name} to SDD in {compile_sdd_time_end:.1f} secs")
                # with open(f'results/compile_time_sdd.txt', 'a') as f:
                #     f.write(f"{data_name} & {compile_sdd_time_end:.1f}\n")
            elif args[2] == '-ddnnf':
                compile_ddnnf_time_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                           resource.getrusage(resource.RUSAGE_SELF).ru_utime
                rodt_to_ddnnf(data, dt_save, None)
                compile_ddnnf_time_end = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                         resource.getrusage(resource.RUSAGE_SELF).ru_utime - compile_ddnnf_time_start
                print(f"compile {data_name} to d-DNNF in {compile_ddnnf_time_end:.1f} secs")
                # with open(f'results/compile_time_ddnnf.txt', 'a') as f:
                #     f.write(f"{data_name} & {compile_ddnnf_time_end:.1f}\n")
