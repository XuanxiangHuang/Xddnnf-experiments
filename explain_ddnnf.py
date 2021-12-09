#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   experiment: Explaining d-DNNF Classifiers
#   Author: Xuanxiang Huang
#
################################################################################
import resource
import sys
from math import ceil
from src.decision_tree import *
from src.ddnnf import dDNNF
from src.explain.xpddnnf import XpdDnnf
################################################################################


def explain_ddnnf(dataset, ddnnf_file, dt_file, save_dir):
    file = dataset.split(sep='/')[-1]
    name = file.split(sep='.')[0]
    dt = DecisionTree.from_file(dt_file)
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

    for i in range(d_len):
        time_i_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                       resource.getrusage(resource.RUSAGE_SELF).ru_utime
        xpddnnf = XpdDnnf(ddnnf, verb=0)
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
            dt_file = f"models/dts/binary/{data_name}.pkl"
            # save xpg/explanations
            save_dir = f"results/"
            # d-DNNF models
            ddnnf_model = f"models/ddnnfs/{data_name}.pkl"
            # explanations
            explain_ddnnf(data, ddnnf_model, dt_file, None)
