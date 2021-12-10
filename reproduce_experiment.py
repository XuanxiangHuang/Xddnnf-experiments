#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Reproduce experiments
#   Author: Xuanxiang Huang
#
################################################################################
from __future__ import print_function
import resource,sys
from math import ceil
from pysdd.sdd import Vtree, SddManager
import decision_tree as DT
import decision_tree_ohe as DT_OHE
# SDD/d-DNNF
from ddnnf import SDD, dDNNF
from ddnnf_ohe import SDD as SDD_OHE
from ddnnf_ohe import dDNNF as dDNNF_OHE
from xpsdd import XpSdd,XpSdd_ohe
from xpddnnf import XpdDnnf,XpdDnnf_ohe
# Anchor
import numpy as np
import pandas as pd
from anchor import anchor_tabular
from Orange.data import Table
################################################################################


def explain_sdd(sdd_file, vtree_file, dataset, dt_file, save_dir):
    ###########################################
    def support_vars(sdd: SddManager):
        """
            Given a SDD manager, return support variables,
            i.e. variables that used/referenced by SDD node.
            :param sdd: SDD manager
            :return:
        """
        all_vars = [_ for _ in sdd.vars]
        _nv = len(all_vars)
        _sup_vars = [None] * _nv

        for i in range(_nv):
            lit = all_vars[i].literal
            assert (lit == i + 1)
            neglit = -all_vars[i].literal
            if sdd.is_var_used(lit) or sdd.is_var_used(neglit):
                _sup_vars[i] = all_vars[i]
        return _sup_vars

    ###########################################
    file = dataset.split(sep='/')[-1]
    name = file.split(sep='.')[0]
    ######################  Pre-processing #####################
    # string to bytes
    sdd_file = bytes(sdd_file, 'utf-8')
    vtree_file = bytes(vtree_file, 'utf-8')
    vtree = Vtree.from_file(vtree_file)
    sdd = SddManager.from_vtree(vtree)
    # Disable gc and minimization
    sdd.auto_gc_and_minimize_off()
    root = sdd.read_sdd_file(sdd_file)
    # obtain all variables (don't cared variables are None)
    tmp_sup_vars = support_vars(sdd)
    tmp_nv = len(tmp_sup_vars)
    # get all features
    dt = DT.DecisionTree.from_file(dt_file)
    tmp_features = [f.replace(' ', '-') for f in dt.features]
    assert (len(tmp_features) == tmp_nv)

    # extract cared features and variables
    sup_vars = []
    features = []
    for jj in range(tmp_nv):
        if tmp_sup_vars[jj]:
            sup_vars.append(tmp_sup_vars[jj])
            features.append(tmp_features[jj])
    nv = len(sup_vars)
    assert (len(features) == nv)
    ######################  Pre-processing #####################
    # enumerate and check
    total_time = 0
    max_time = 0
    min_time = 99999
    max_num_axps = 0
    min_num_axps = 99999
    max_num_cxps = 0
    min_num_cxps = 99999
    all_num_axps = 0
    all_num_cxps = 0

    max_len_axps = 0
    min_len_axps = 99999
    max_len_cxps = 0
    min_len_cxps = 99999
    all_len_axps = 0
    all_len_cxps = 0
    all_exps = 0

    # test data
    d_len = int(len(dt.X_test) / 2)
    # all data
    # d_len = len(dt.datatable.X)

    xpsdd = XpSdd(root, nv, features, sup_vars, verb=0)

    for i in range(d_len):
        # extract value of cared features
        # test data
        tmp_sample = list(dt.X_test[i])
        # all data
        # tmp_sample = list(dt.datatable.X[i])

        sample = []
        for ii in range(tmp_nv):
            if tmp_sup_vars[ii]:
                sample.append(tmp_sample[ii])

        lits = xpsdd.inst2lits(inst=sample)

        time_i_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                       resource.getrusage(resource.RUSAGE_SELF).ru_utime

        axps, cxps = xpsdd.enum_exps(lits)

        time_i = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                 resource.getrusage(resource.RUSAGE_SELF).ru_utime - time_i_start
        total_time += time_i

        num_axps = len(axps)
        num_cxps = len(cxps)
        l_a = [len(x) for x in axps]
        l_c = [len(x) for x in cxps]
        all_len_axps += sum(l_a)
        all_len_cxps += sum(l_c)
        all_num_axps += num_axps
        all_num_cxps += num_cxps
        all_exps += (num_axps + num_cxps)

        if max_num_axps < num_axps:
            max_num_axps = num_axps
        if min_num_axps > num_axps:
            min_num_axps = num_axps
        if max_num_cxps < num_cxps:
            max_num_cxps = num_cxps
        if min_num_cxps > num_cxps:
            min_num_cxps = num_cxps

        if max_len_axps < max(l_a):
            max_len_axps = max(l_a)
        if min_len_axps > min(l_a):
            min_len_axps = min(l_a)
        if max_len_cxps < max(l_c):
            max_len_cxps = max(l_c)
        if min_len_cxps > min(l_c):
            min_len_cxps = min(l_c)

        axps_save = []
        cxps_save = []

        for tmp in axps:
            feats_output = [features[i] for i in tmp]
            axps_save.append(feats_output)
        for tmp in cxps:
            feats_output = [features[i] for i in tmp]
            cxps_save.append(feats_output)

        if time_i > max_time:
            max_time = time_i
        if time_i < min_time:
            min_time = time_i

    results = f"\n{name} & "
    results += f"{tmp_nv} & {d_len} & "
    results += f" & "
    results += f" & "
    results += f"{round(all_exps / d_len):.0f} & "
    results += f"{max_num_axps} & {min_num_axps} & {round(all_num_axps / d_len):.0f} & "
    results += f"{ceil((all_len_axps / all_num_axps) / tmp_nv * 100):.0f} & "
    results += f"{max_num_cxps} & {min_num_cxps} & {round(all_num_cxps / d_len):.0f} & "
    results += f"{ceil((all_len_cxps / all_num_cxps) / tmp_nv * 100):.0f} & "
    results += "{0:.3f} & {1:.3f} & {2:.3f} & {3:.3f}" \
        .format(total_time, max_time, min_time, total_time / d_len)

    print(results)

    if save_dir:
        with open(f'{save_dir}/aaai22_sdd.txt', 'a') as f:
            f.write(results)


def explain_sdd_ohe(dataset, dt_file, save_dir):
    file = dataset.split(sep='/')[-1]
    name = file.split(sep='.')[0]
    ################### Compile ###################
    dt = DT_OHE.DecisionTree.from_file(dt_file)
    dt_train, dt_test = dt.train()
    in_model = dt.dt2bnet()
    Sdd = SDD_OHE()

    Sdd.compile(in_model, dt.features_bin, dt.features,
                dt.feat_to_binarize, dt.feat_to_original)

    acc_train = round(Sdd.accuracy(dt.train_datatable_bin.X, dt.train_datatable_bin.Y), 3)
    acc_test = round(Sdd.accuracy(dt.test_datatable_bin.X, dt.test_datatable_bin.Y), 3)

    print(f"DT, Train accuracy: {dt_train * 100.0}%")
    print(f"DT, Test accuracy: {dt_test * 100.0}%")

    print(f"SDD, Train accuracy: {acc_train * 100.0}%")
    print(f"SDD, Test accuracy: {acc_test * 100.0}%")

    if comp_mlmodel_sdd(dt, Sdd, dt.train_datatable_bin.X, dt.test_datatable_bin.X):
        print('DT, SDD are not consistent')
        return None
    ################### Compile ###################
    features = [f.replace(' ', '-') for f in dt.features]
    print(dt.features)
    print(dt.features_bin)
    print(features)

    # enumerate and check
    total_time = 0
    max_time = 0
    min_time = 99999
    max_num_axps = 0
    min_num_axps = 99999
    max_num_cxps = 0
    min_num_cxps = 99999
    all_num_axps = 0
    all_num_cxps = 0

    max_len_axps = 0
    min_len_axps = 99999
    max_len_cxps = 0
    min_len_cxps = 99999
    all_len_axps = 0
    all_len_cxps = 0
    all_exps = 0

    # test data
    d_len = int(len(dt.test_datatable.X) / 2)
    # train data
    # d_len = len(dt.train_datatable.X)

    xpsdd = XpSdd_ohe(Sdd.root, Sdd.b_nv, Sdd.o_nv, Sdd.b_feats, Sdd.o_feats,
                      Sdd.sup_b_vars, Sdd.sup_o_vars, Sdd.to_b_feats, Sdd.to_o_feats, verb=0)

    for i in range(d_len):
        # extract value of cared features
        # test data
        tmp_sample = list(dt.test_datatable_bin.X[i])
        # train data
        # tmp_sample = list(dt.train_datatable_bin.X[i])

        sample = [int(pt) for pt in tmp_sample]
        lits = xpsdd.inst2lits(inst=sample)

        time_i_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                       resource.getrusage(resource.RUSAGE_SELF).ru_utime

        axps, cxps = xpsdd.enum_exps(lits)

        time_i = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                 resource.getrusage(resource.RUSAGE_SELF).ru_utime - time_i_start
        total_time += time_i

        # check each axp and cxp
        for item in axps:
            assert xpsdd.check_one_axp(lits, item)
        for item in cxps:
            assert xpsdd.check_one_cxp(lits, item)

        # assert axps are MHS of cxps, vice versa
        assert checkMHS(axps, cxps)

        num_axps = len(axps)
        num_cxps = len(cxps)
        l_a = [len(x) for x in axps]
        l_c = [len(x) for x in cxps]
        all_len_axps += sum(l_a)
        all_len_cxps += sum(l_c)
        all_num_axps += num_axps
        all_num_cxps += num_cxps
        all_exps += (num_axps + num_cxps)

        if max_num_axps < num_axps:
            max_num_axps = num_axps
        if min_num_axps > num_axps:
            min_num_axps = num_axps
        if max_num_cxps < num_cxps:
            max_num_cxps = num_cxps
        if min_num_cxps > num_cxps:
            min_num_cxps = num_cxps

        if max_len_axps < max(l_a):
            max_len_axps = max(l_a)
        if min_len_axps > min(l_a):
            min_len_axps = min(l_a)
        if max_len_cxps < max(l_c):
            max_len_cxps = max(l_c)
        if min_len_cxps > min(l_c):
            min_len_cxps = min(l_c)

        axps_save = []
        cxps_save = []

        for tmp in axps:
            feats_output = [features[i] for i in tmp]
            axps_save.append(feats_output)
        for tmp in cxps:
            feats_output = [features[i] for i in tmp]
            cxps_save.append(feats_output)

        if time_i > max_time:
            max_time = time_i
        if time_i < min_time:
            min_time = time_i

    results = f"\n{name} & "
    results += f"{Sdd.o_nv} & {d_len} & "
    results += f"{acc_test * 100.0:.1f} & "
    results += f" & "
    results += f"{round(all_exps / d_len):.0f} & "
    results += f"{max_num_axps} & {min_num_axps} & {round(all_num_axps / d_len):.0f} & "
    results += f"{ceil((all_len_axps / all_num_axps) / Sdd.o_nv * 100):.0f} & "
    results += f"{max_num_cxps} & {min_num_cxps} & {round(all_num_cxps / d_len):.0f} & "
    results += f"{ceil((all_len_cxps / all_num_cxps) / Sdd.o_nv * 100):.0f} & "
    results += "{0:.3f} & {1:.3f} & {2:.3f} & {3:.3f}" \
        .format(total_time, max_time, min_time, total_time / d_len)

    print(results)

    if save_dir:
        with open(f'{save_dir}/aaai22_sdd.txt', 'a') as f:
            f.write(results)


def checkMHS(in_axps: list, in_cxps: list):
    # given a list of axp and a list of cxp,
    # check if they are minimal-hitting-set (MHS) of each other
    # 1. uniqueness, and no subset(superset) exists;
    if not in_axps or not in_cxps:
        print(f"input empty: {in_axps}, {in_cxps}")
        return False
    axps = sorted(in_axps, key=lambda x: len(x))
    axps_ = axps[:]
    while axps:
        axp = axps.pop()
        set_axp = set(axp)
        for ele in axps:
            set_ele = set(ele)
            if set_axp.issuperset(set_ele) or set_axp.issubset(set_ele):
                print(f"axp is not unique: {set_axp}, {set_ele}")
                return False
    cxps = sorted(in_cxps, key=lambda x: len(x))
    cxps_ = cxps[:]
    while cxps:
        cxp = cxps.pop()
        set_cxp = set(cxp)
        for ele in cxps:
            set_ele = set(ele)
            if set_cxp.issuperset(set_ele) or set_cxp.issubset(set_ele):
                print(f"cxp is not unique: {set_cxp}, {set_ele}")
                return False
    # 2. minimal hitting set;
    for axp in axps_:
        set_axp = set(axp)
        for cxp in cxps_:
            set_cxp = set(cxp)
            if not (set_axp & set_cxp):  # not a hitting set
                print(f"not a hitting set: axp:{set_axp}, cxp:{set_cxp}")
                return False
    # axp is a MHS of cxps
    for axp in axps_:
        set_axp = set(axp)
        for ele in set_axp:
            tmp = set_axp - {ele}
            size = len(cxps_)
            for cxp in cxps_:
                set_cxp = set(cxp)
                if tmp & set_cxp:
                    size -= 1
            if size == 0:  # not minimal
                print(f"axp is not minimal hitting set: "
                      f"axp {set_axp} covers #{len(cxps_)}, "
                      f"its subset {tmp} covers #{len(cxps_) - size}, "
                      f"so {ele} is redundant")
                return False
    # cxp is a MHS of axps
    for cxp in cxps_:
        set_cxp = set(cxp)
        for ele in set_cxp:
            tmp = set_cxp - {ele}
            size = len(axps_)
            for axp in axps_:
                set_axp = set(axp)
                if tmp & set_axp:
                    size -= 1
            if size == 0:
                print(f"cxp is not minimal hitting set: "
                      f"cxp {set_cxp} covers #{len(axps_)}, "
                      f"its subset {tmp} covers #{len(axps_) - size}, "
                      f"so {ele} is redundant")
                return False
    return True


def comp_mlmodel_sdd(model, sdd, X_train, X_test):
    total = len(X_train)
    count = 0
    collect = []
    for i in X_train:
        pred1, pred2 = model.classifier(i), sdd.predict_one(i)
        if pred1 != pred2:
            print(f'train: {i}, MLmodel: {pred1}, bdd: {pred2}')
            collect.append(i)
            count += 1

    for i in X_test:
        pred1, pred2 = model.classifier(i), sdd.predict_one(i)
        if pred1 != pred2:
            print(f'test: {i}, MLmodel: {pred1}, bdd: {pred2}')
            collect.append(i)
            count += 1
    return count


def comp_mlmodel_ddnnf(model, ddnnf, X_train, X_test):
    count = 0
    collect = []
    for i in X_train:
        pred1, pred2 = model.classifier(i), ddnnf.predict_one(i)
        if pred1 != pred2:
            print(f'train: {i}, MLmodel: {pred1}, dDNNF: {pred2}')
            collect.append(i)
            count += 1

    for i in X_test:
        pred1, pred2 = model.classifier(i), ddnnf.predict_one(i)
        if pred1 != pred2:
            print(f'test: {i}, MLmodel: {pred1}, dDNNF: {pred2}')
            collect.append(i)
            count += 1
    return count


def explain_ddnnf(dataset, ddnnf_file, dt_file, save_dir):
    file = dataset.split(sep='/')[-1]
    name = file.split(sep='.')[0]
    dt = DT.DecisionTree.from_file(dt_file)
    ddnnf = dDNNF.from_file(ddnnf_file)
    tmp_features = [f.replace(' ', '-') for f in dt.features]

    # enumerate and check
    total_time = 0
    max_time = 0
    min_time = 99999
    max_num_axps = 0
    min_num_axps = 99999
    max_num_cxps = 0
    min_num_cxps = 99999
    all_num_axps = 0
    all_num_cxps = 0

    max_len_axps = 0
    min_len_axps = 99999
    max_len_cxps = 0
    min_len_cxps = 99999
    all_len_axps = 0
    all_len_cxps = 0
    all_exps = 0

    # test data
    d_len = int(len(dt.X_test) / 2)
    # all data
    # d_len = len(dt.datatable.X)

    xpddnnf = XpdDnnf(ddnnf, verb=0)

    for i in range(d_len):
        time_i_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                       resource.getrusage(resource.RUSAGE_SELF).ru_utime
        # extract value of cared features
        # test data
        tmp_sample = list(dt.X_test[i])
        # all data
        # tmp_sample = list(dt.datatable.X[i])

        sample = []
        for ii in range(len(tmp_sample)):
            if tmp_features[ii] in ddnnf.sup_feats:
                sample.append(tmp_sample[ii])
        # get prediction
        pred = ddnnf.predict_one(sample)
        axps, cxps = xpddnnf.enum_exps(sample, pred)

        time_i = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                 resource.getrusage(resource.RUSAGE_SELF).ru_utime - time_i_start
        total_time += time_i

        num_axps = len(axps)
        num_cxps = len(cxps)
        l_a = [len(x) for x in axps]
        l_c = [len(x) for x in cxps]
        all_len_axps += sum(l_a)
        all_len_cxps += sum(l_c)
        all_num_axps += num_axps
        all_num_cxps += num_cxps
        all_exps += (num_axps + num_cxps)

        if max_num_axps < num_axps:
            max_num_axps = num_axps
        if min_num_axps > num_axps:
            min_num_axps = num_axps
        if max_num_cxps < num_cxps:
            max_num_cxps = num_cxps
        if min_num_cxps > num_cxps:
            min_num_cxps = num_cxps

        if max_len_axps < max(l_a):
            max_len_axps = max(l_a)
        if min_len_axps > min(l_a):
            min_len_axps = min(l_a)
        if max_len_cxps < max(l_c):
            max_len_cxps = max(l_c)
        if min_len_cxps > min(l_c):
            min_len_cxps = min(l_c)

        axps_save = []
        cxps_save = []

        for tmp in axps:
            feats_output = [ddnnf.sup_feats[i] for i in tmp]
            axps_save.append(feats_output)
        for tmp in cxps:
            feats_output = [ddnnf.sup_feats[i] for i in tmp]
            cxps_save.append(feats_output)

        if time_i > max_time:
            max_time = time_i
        if time_i < min_time:
            min_time = time_i

    results = f"\n{name} & "
    results += f"{len(ddnnf.features)} & {d_len} & "
    results += f" & "
    results += f"{ddnnf.size(ddnnf.root)} & "
    results += f"{round(all_exps / d_len):.0f} & "
    results += f"{max_num_axps} & {min_num_axps} & {round(all_num_axps / d_len):.0f} & "
    results += f"{ceil((all_len_axps / all_num_axps) / len(ddnnf.features) * 100):.0f} & "
    results += f"{max_num_cxps} & {min_num_cxps} & {round(all_num_cxps / d_len):.0f} & "
    results += f"{ceil((all_len_cxps / all_num_cxps) / len(ddnnf.features) * 100):.0f} & "
    results += "{0:.3f} & {1:.3f} & {2:.3f} & {3:.3f}" \
        .format(total_time, max_time, min_time, total_time / d_len)

    print(results)

    if save_dir:
        with open(f'{save_dir}/aaai22_ddnnf.txt', 'a') as f:
            f.write(results)


def explain_ddnnf_ohe(dataset, dt_file, save_dir):
    file = dataset.split(sep='/')[-1]
    name = file.split(sep='.')[0]
    dt = DT_OHE.DecisionTree.from_file(dt_file)
    dt_train, dt_test = dt.train()
    ddnnf = dDNNF_OHE()
    ddnnf.compile(dt, dt.features_bin, dt.features,
                  dt.feat_to_binarize, dt.feat_to_original)

    acc_train = round(ddnnf.accuracy(dt.train_datatable_bin.X, dt.train_datatable_bin.Y), 3)
    acc_test = round(ddnnf.accuracy(dt.test_datatable_bin.X, dt.test_datatable_bin.Y), 3)

    print(f"DT, Train accuracy: {dt_train * 100.0:.1f}%")
    print(f"DT, Test accuracy: {dt_test * 100.0:.1f}%")

    print(f"d-DNNF, Train accuracy: {acc_train * 100.0:.1f}%")
    print(f"d-DNNF, Test accuracy: {acc_test * 100.0:.1f}%")

    if comp_mlmodel_ddnnf(dt, ddnnf, dt.train_datatable_bin.X, dt.test_datatable_bin.X):
        print('DT, d-DNNF are not consistent')
        return None

    #######################################
    features = [f.replace(' ', '-') for f in ddnnf.o_feats]
    print(ddnnf.o_feats)
    print(ddnnf.b_feats)
    print(features)

    # enumerate and check
    total_time = 0
    max_time = 0
    min_time = 99999
    max_num_axps = 0
    min_num_axps = 99999
    max_num_cxps = 0
    min_num_cxps = 99999
    all_num_axps = 0
    all_num_cxps = 0

    max_len_axps = 0
    min_len_axps = 99999
    max_len_cxps = 0
    min_len_cxps = 99999
    all_len_axps = 0
    all_len_cxps = 0
    all_exps = 0

    # test data
    d_len = int(len(dt.test_datatable.X) / 2)
    # train data
    # d_len = len(dt.train_datatable.X)

    xpddnnf = XpdDnnf_ohe(ddnnf, verb=0)

    for i in range(d_len):
        time_i_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                       resource.getrusage(resource.RUSAGE_SELF).ru_utime
        # extract value of cared features
        # test data
        tmp_sample = list(dt.test_datatable_bin.X[i])
        # train data
        # tmp_sample = list(dt.train_datatable_bin.X[i])

        sample = [int(pt) for pt in tmp_sample]
        # get prediction
        pred = ddnnf.predict_one(sample)
        axps, cxps = xpddnnf.enum_exps(sample, pred)

        time_i = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                 resource.getrusage(resource.RUSAGE_SELF).ru_utime - time_i_start
        total_time += time_i

        # check each axp and cxp
        for item in axps:
            assert xpddnnf.check_one_axp(sample, pred, item)
        for item in cxps:
            assert xpddnnf.check_one_cxp(sample, pred, item)

        # assert axps are MHS of cxps, vice versa
        assert checkMHS(axps, cxps)

        num_axps = len(axps)
        num_cxps = len(cxps)
        l_a = [len(x) for x in axps]
        l_c = [len(x) for x in cxps]
        all_len_axps += sum(l_a)
        all_len_cxps += sum(l_c)
        all_num_axps += num_axps
        all_num_cxps += num_cxps
        all_exps += (num_axps + num_cxps)

        if max_num_axps < num_axps:
            max_num_axps = num_axps
        if min_num_axps > num_axps:
            min_num_axps = num_axps
        if max_num_cxps < num_cxps:
            max_num_cxps = num_cxps
        if min_num_cxps > num_cxps:
            min_num_cxps = num_cxps

        if max_len_axps < max(l_a):
            max_len_axps = max(l_a)
        if min_len_axps > min(l_a):
            min_len_axps = min(l_a)
        if max_len_cxps < max(l_c):
            max_len_cxps = max(l_c)
        if min_len_cxps > min(l_c):
            min_len_cxps = min(l_c)

        axps_save = []
        cxps_save = []

        for tmp in axps:
            feats_output = [features[i] for i in tmp]
            axps_save.append(feats_output)
        for tmp in cxps:
            feats_output = [features[i] for i in tmp]
            cxps_save.append(feats_output)

        if time_i > max_time:
            max_time = time_i
        if time_i < min_time:
            min_time = time_i

    results = f"\n{name} & "
    results += f"{ddnnf.o_nv} & {d_len} & "
    results += f"{acc_test * 100.0:.1f} & "
    results += f"{ddnnf.size(ddnnf.root)} & "
    results += f"{round(all_exps / d_len):.0f} & "
    results += f"{max_num_axps} & {min_num_axps} & {round(all_num_axps / d_len):.0f} & "
    results += f"{ceil((all_len_axps / all_num_axps) / len(features) * 100):.0f} & "
    results += f"{max_num_cxps} & {min_num_cxps} & {round(all_num_cxps / d_len):.0f} & "
    results += f"{ceil((all_len_cxps / all_num_cxps) / len(features) * 100):.0f} & "
    results += "{0:.3f} & {1:.3f} & {2:.3f} & {3:.3f}" \
        .format(total_time, max_time, min_time, total_time / d_len)

    print(results)

    if save_dir:
        with open(f'{save_dir}/aaai22_ddnnf.txt', 'a') as f:
            f.write(results)


def anchor_call(model, inst, class_names, feature_names, train_data,
                threshold=0.95, verbose=0):
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

def binarize(dt, samples, preds):
    inst_datatable = Table.from_numpy(dt.datatable.domain, samples, preds)
    inst_datatable_bin = dt.continuizer(inst_datatable)
    return inst_datatable_bin.X


def anchor_call_ohe(model, inst, dt, threshold=0.95, verbose=0):
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


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 4 and args[0] == '-bench':
        bench_name = args[1]

        with open(f"{bench_name}", 'r') as fp:
            name_list = fp.readlines()

        for item in name_list:
            # dataset and dt
            data_name = item.strip()
            print(f"############ {data_name} ############")
            data = f"datasets/{data_name}.csv"
            if args[2] == '-sdd':
                if args[3] == '-ohe':
                    dt_file = f"models/dts/categorical/{data_name}.pkl"
                    save_dir = f"tmp/"
                    explain_sdd_ohe(data, dt_file, save_dir)
                else:
                    dt_file = f"models/dts/binary/{data_name}.pkl"
                    sdd_save = f"models/sdds/{data_name}.txt"
                    vtree_save = f"models/sdds/{data_name}_vtree.txt"
                    save_dir = f"tmp/"
                    explain_sdd(sdd_save, vtree_save, data, dt_file, save_dir)
            elif args[2] == '-ddnnf':
                if args[3] == '-ohe':
                    dt_file = f"models/dts/categorical/{data_name}.pkl"
                    save_dir = f"tmp/"
                    explain_ddnnf_ohe(data, dt_file, save_dir)
                else:
                    dt_file = f"models/dts/binary/{data_name}.pkl"
                    save_dir = f"tmp/"
                    ddnnf_model = f"models/ddnnfs/{data_name}.pkl"
                    explain_ddnnf(data, ddnnf_model, dt_file, save_dir)
            elif args[2] == '-anchor':
                if args[3] == '-ohe':
                    dt_file = f"models/dts/categorical/{data_name}.pkl"
                    dt = DT_OHE.DecisionTree.from_file(dt_file)
                    dt_train, dt_test = dt.train()
                    ddnnf = dDNNF_OHE()
                    ddnnf.compile(dt, dt.features_bin, dt.features,
                                  dt.feat_to_binarize, dt.feat_to_original)
                    print(dt.features)
                    print(dt.features_bin)
                    exps = []
                    atimes = []
                    d_len = int(len(dt.test_datatable.X)/2)
                    for i in range(d_len):
                        print(f"instance {i} of {d_len}:")
                        sample = dt.test_datatable.X[i]
                        exp, time = anchor_call_ohe(ddnnf, sample, dt, verbose=2)
                        exps.append(exp)
                        atimes.append(time)

                    assert len(exps) == d_len
                    results = f"{data_name} & {d_len} & {ceil((sum(exps) / d_len) / len(dt.features) * 100):.0f} & "
                    results += "{0:.3f} & {1:.3f} & {2:.3f} & {3:.3f}\n".format(sum(atimes), max(atimes), min(atimes),
                                                                                sum(atimes) / len(atimes))
                    # with open('results/aaai22_anchor.txt', 'a') as f:
                    #     f.write(results)
                    print(results)
                else:
                    dt_file = f"models/dts/binary/{data_name}.pkl"
                    dt = DT.DecisionTree.from_file(dt_file)
                    dt_train, dt_test = dt.train()
                    ddnnf = dDNNF()
                    ddnnf.compile(dt)
                    exps = []
                    atimes = []
                    d_len = int(len(dt.X_test)/2)
                    for i in range(d_len):
                        print(f"instance {i} of {d_len}:")
                        sample = dt.X_test[i]
                        exp, time = anchor_call(ddnnf, sample, list(dt.classes), dt.features, dt.X_train, verbose=2)
                        exps.append(exp)
                        atimes.append(time)

                    assert len(exps) == d_len
                    results = f"{data_name} & {d_len} & {ceil((sum(exps) / d_len) / len(dt.features) * 100):.0f} & "
                    results += "{0:.3f} & {1:.3f} & {2:.3f} & {3:.3f}\n".format(sum(atimes), max(atimes), min(atimes),
                                                                                sum(atimes) / len(atimes))
                    # with open('results/aaai22_anchor.txt', 'a') as f:
                    #     f.write(results)
                    print(results)
