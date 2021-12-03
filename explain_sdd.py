#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   experiment: Explaining SDD Classifiers
#   Author: Xuanxiang Huang
#
################################################################################
import resource
import sys
from math import ceil
from decision_tree import *
from pysdd.sdd import Vtree, SddManager
from xpsdd import XpSdd
################################################################################


def support_vars(sdd: SddManager):
    """
        Given a SDD manager, return support variables,
        i.e. variables that used/referenced by SDD node.
        :param sdd: SDD manager
        :return:
    """
    all_vars = [_ for _ in sdd.vars]
    nv = len(all_vars)
    sup_vars = [None] * nv

    for i in range(nv):
        lit = all_vars[i].literal
        assert (lit == i + 1)
        neglit = -all_vars[i].literal
        if sdd.is_var_used(lit) or sdd.is_var_used(neglit):
            sup_vars[i] = all_vars[i]
    return sup_vars


def explain_sdd(sdd_file, vtree_file, dataset, dt_file, save_dir):
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
    dt = DecisionTree.from_file(dt_file)
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

    for i in range(d_len):

        xpsdd = XpSdd(root, nv, features, sup_vars, verb=0)
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
            # sdd and vtree
            sdd_save = f"sdd_models/{data_name}.txt"
            vtree_save = f"sdd_models/{data_name}_vtree.txt"
            # save xpg/explanations
            save_dir = f"results/"
            # explanations
            explain_sdd(sdd_save, vtree_save, data, dt_file, None)
