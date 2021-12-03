#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   experiment: Explaining d-DNNF Classifiers (one-hot-encoding)
#   Author: Xuanxiang Huang
#
################################################################################
import resource
import sys
from math import ceil
from decision_tree_ohe import DecisionTree
from ddnnf_ohe import dDNNF
from xpddnnf_ohe import XpdDnnf
################################################################################


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


def explain_ddnnf(dataset, dt_file, save_dir):
    file = dataset.split(sep='/')[-1]
    name = file.split(sep='.')[0]
    dt = DecisionTree.from_file(dt_file)
    dt_train, dt_test = dt.train()
    ddnnf = dDNNF()
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

    for i in range(d_len):
        time_i_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                       resource.getrusage(resource.RUSAGE_SELF).ru_utime
        xpddnnf = XpdDnnf(ddnnf, verb=0)
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
            # save xpg/explanations
            save_dir = f"results/"
            # explanations
            explain_ddnnf(data, dt_file, None)
