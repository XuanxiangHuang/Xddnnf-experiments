#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Check Read-Once DT
#   Author: Xuanxiang Huang
#
################################################################################
import sys
import decision_tree as DT
import decision_tree_ohe as DTOHE
################################################################################


def check_rodt(dt_file):
    dt = DT.DecisionTree.from_file(dt_file)
    dt_train, dt_test = dt.train()
    print(f"DT, Train accuracy: {dt_train * 100.0}%")
    print(f"DT, Test accuracy: {dt_test * 100.0}%")
    dt.check_read_once()
    print(f"DT size: {dt.size()}")


def check_rodt_ohe(dt_file):
    dt = DTOHE.DecisionTree.from_file(dt_file)
    dt_train, dt_test = dt.train()
    print(f"DT, Train accuracy: {dt_train * 100.0}%")
    print(f"DT, Test accuracy: {dt_test * 100.0}%")
    dt.check_read_once()
    print(f"DT size: {dt.size()}")


# -dataset xxx -cat
if __name__ == '__main__':
    args = sys.argv[1:]
    if (len(args) == 2 or len(args) == 3) and args[0] == '-bench':
        bench_name = args[1]

        with open(f"{bench_name}", 'r') as fp:
            name_list = fp.readlines()

        for item in name_list:
            data_name = item.strip()
            # dataset and dt model
            if len(args) == 3 and args[2] == '-cat':
                dt_save_ohe = f"../models/dts/categorical/{data_name}.pkl"
                print(f"############ Categorical: {data_name} ############")
                check_rodt_ohe(dt_save_ohe)
            else:
                dt_save = f"../models/dts/binary/{data_name}.pkl"
                print(f"############ Binary: {data_name} ############")
                check_rodt(dt_save)
