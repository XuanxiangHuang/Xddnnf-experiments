#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Training DT models
#   Author: Xuanxiang Huang
#
################################################################################
import decision_tree as DT
import decision_tree_ohe as DTOHE
################################################################################


def train_dt(dataset, max_depth, train_threshold, test_threshold, save_name):
    dt = DT.DecisionTree(dataset, max_depth=max_depth)
    acc_train, acc_test = dt.train()

    if acc_train < train_threshold or acc_test < test_threshold:
        print(f'DT: train accuracy {acc_train} < {train_threshold}'
              f' or test accuracy {acc_test} < {test_threshold}')
        return
    else:
        print(f"DT, Train accuracy: {acc_train * 100.0}%")
        print(f"DT, Test accuracy: {acc_test * 100.0}%")

    if save_name:
        dt.save_model(dt, save_name)

    dt.check_read_once()
    print(f"features: {dt.features}")
    print(f"classes: {dt.classes}")
    print(dt)


def train_dt_ohe(dataset, max_depth, train_threshold, test_threshold, save_name):
    dt = DTOHE.DecisionTree(dataset, max_depth=max_depth)
    acc_train, acc_test = dt.train()

    if acc_train < train_threshold or acc_test < test_threshold:
        print(f'DT: train accuracy {acc_train} < {train_threshold}'
              f' or test accuracy {acc_test} < {test_threshold}')
        return
    else:
        print(f"DT, Train accuracy: {acc_train * 100.0}%")
        print(f"DT, Test accuracy: {acc_test * 100.0}%")

    if save_name:
        dt.save_model(dt, save_name)

    dt.check_read_once()
    print(f"features: {dt.features}")
    print(f"classes: {dt.classes}")
    print(f"features (binarized): {dt.features_bin}")
    print(f"classes (binarized): {dt.classes_bin}")
    print(dt)


##########################################################################################
if __name__ == '__main__':
    name = "corral"
    data = f"datasets/{name}.csv"
    dt_save = f"{name}.pkl"

    name_ohe = "vote"
    data_ohe = f"datasets/{name_ohe}.csv"
    dt_save_ohe = f"{name_ohe}.pkl"

    train_t = 0.75
    test_t = 0.7

    train_dt(data, 8, train_t, test_t, None)
    train_dt_ohe(data_ohe, 8, train_t, test_t, None)
