#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Training
#   Author: Xuanxiang Huang
#
################################################################################
from __future__ import print_function
import resource,sys,os,getopt
import decision_tree as DT
import decision_tree_ohe as DT_OHE
# SDD/d-DNNF
from ddnnf import SDD, dDNNF
from ddnnf_ohe import SDD as SDD_OHE
from ddnnf_ohe import dDNNF as dDNNF_OHE
################################################################################


class Options(object):
    """
        Class for representing command-line options.
    """

    def __init__(self, command):
        self.command = command
        self.reproduce = ''
        self.max_depth = 6
        self.test_split = 0.2
        self.train_classifier = 'dt'
        self.train_threshold = 0.75
        self.test_threshold = 0.7
        self.ohe = False
        self.input_dt = ''
        self.save = ''
        self.dataset = None

        if command:
            self.parse(command)

    def parse(self, command):
        """
            Parser.
        """
        try:
            opts, args = getopt.getopt(command[1:], 'c:d:D:f:l:s:t:',
                                       ['classifier=',
                                        'depth=',
                                        'dataset=',
                                        'from=',
                                        'help',
                                        'learn=',
                                        'ohe',
                                        'save=',
                                        'test-split=',
                                        'reproduce='
                                        ])
        except getopt.GetoptError as err:
            sys.stderr.write(str(err).capitalize())
            self.usage()
            sys.exit(1)

        for opt, arg in opts:
            if opt in ('-d', '--depth'):
                max_depth = int(arg)
                if max_depth <= 0:
                    print('wrong parameter: -d (depth)')
                    sys.exit(1)
                self.max_depth = max_depth
            elif opt in ('-c', '--classifier'):
                train_classifier = str(arg)
                if train_classifier not in ('dt', 'sdd', 'ddnnf'):
                    print('wrong parameter: -c (classifier)')
                    sys.exit(1)
                self.train_classifier = train_classifier
            elif opt in ('-t', '--test-split'):
                test_split = float(arg)
                if test_split > 1.0 or test_split < 0.2:
                    print('wrong parameter: -t (test-split)')
                    sys.exit(1)
                self.test_split = test_split
            elif opt in ('-l', '--learn'):
                thresholds = arg.split(':')
                train_threshold = float(thresholds[0])
                test_threshold = float(thresholds[1])
                if (train_threshold < 0.5 or train_threshold > 1) or \
                        (test_threshold < 0.5 or test_threshold > 1):
                    print('wrong parameter: -l (learn) threshold')
                    sys.exit(1)
                self.train_threshold = train_threshold
                self.test_threshold = test_threshold
            elif opt in ('-f', '--from'):
                self.input_dt = str(arg)
            elif opt in ('-D', '--dataset'):
                self.dataset = arg
            elif opt in '--ohe':
                self.ohe = True
            elif opt in ('-s', '--save'):
                self.save = str(arg)
            elif opt in '--help':
                self.usage()
                sys.exit(0)
            elif opt in '--reproduce':
                self.reproduce = str(arg)
            else:
                assert False, 'unhandled option: {0} {1}'.format(opt, arg)

        if not self.dataset and not len(self.reproduce):
            print('error: no input dataset')
            self.usage()
            sys.exit(1)

    def usage(self):
        print('Usage: ' + os.path.basename(self.command[0]) + ' [options]')
        print('example #1: ./train -d 6 -l 0.8:0.8 -c dt -D datasets/corral.csv -s tmp/dt_corral')
        print('             train sdd classifier (train accuracy >= 0.85, test accuracy >= 0.8')
        print('example #2: ./train -c sdd -D datasets/adult.csv -f models/dts/categorical/adult.pkl --ohe -s tmp/adult')
        print('example #3 (Reproducibility): ./train --reproduce data_cat_list.txt -c sdd --ohe')
        print('Options:')
        print(' -d,  --depth=<int>       Maximal depth of a tree (default 8);')
        print(' -l,  --learn=<float>:<float> Learn dt with threshold. (default: train >= 0.5, test >= 0.5);')
        print('                              no applicable for sdd/ddnnf;')
        print(' -t,  --test-split=<float>    Training and test sets split (default 0.2 ∈ [0.2, 1.0]);')
        print(' -c,  --classifier=<string>   Type of classifier. (default: dt ∈ {dt, sdd, ddnnf});')
        print(' -f,  --from=<string>         Compiling sdd/ddnnf from this input dt model;')
        print(' --ohe,                       Use one-hot-encoding;')
        print(' -s,  --save=<string>         Provide a NAME (without suffix) for saving train classifier;')
        print('                              .pkl for dd and ddnnf, .txt for sdd and its vtree;')
        print(' --help                       Show this message.')

################################################################################


def train_dt(dataset, max_depth, train_threshold, test_threshold, learning_rate, save_name):
    dt = DT.DecisionTree(dataset, max_depth=max_depth, learning_rate=learning_rate)
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


def train_dt_ohe(dataset, max_depth, train_threshold, test_threshold, learning_rate, save_name):
    dt = DT_OHE.DecisionTree(dataset, max_depth=max_depth, learning_rate=learning_rate)
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
        ddnnf.save_model(ddnnf, filename=f'{save_dir}')


def compile_sdd_ohe(dt_file, sdd_file, vtree_file):
    # SDD with OHE
    dt = DT_OHE.DecisionTree.from_file(dt_file)
    dt_train, dt_test = dt.train()
    # cat
    in_model = dt.dt2bnet()
    Sdd = SDD_OHE()
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


def rodt_to_ddnnf_ohe(dataset, dt_file, save_dir):
    # d-DNNF with OHE
    file = dataset.split(sep='/')[-1]
    name = file.split(sep='.')[0]
    dt = DT_OHE.DecisionTree.from_file(dt_file)
    dt_train, dt_test = dt.train()

    # RODT -> d-DNNF
    ddnnf = dDNNF_OHE()
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
        ddnnf.save_model(ddnnf, filename=f'{save_dir}')


##########################################################################################
if __name__ == '__main__':
    """
        Train Decision Trees/d-DNNFs/SDDs:
            d-DNNFs/SDDs are compiled from Decision Trees.
    """
    options = Options(sys.argv)

    if not len(options.reproduce):
        dataset = options.dataset
        basename = os.path.splitext(os.path.basename(options.dataset))[0]
        print(f"### train {basename} ###")

        if options.train_classifier == 'dt':
            if options.ohe:
                train_dt_ohe(options.dataset, options.max_depth,
                             options.train_threshold, options.test_threshold, options.test_split,
                             options.save+'.pkl')
            else:
                train_dt(options.dataset, options.max_depth,
                         options.train_threshold, options.test_threshold, options.test_split,
                         options.save+'.pkl')

        elif options.train_classifier == 'sdd':
            compile_sdd_time_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                     resource.getrusage(resource.RUSAGE_SELF).ru_utime
            if options.ohe:
                compile_sdd_ohe(options.input_dt, options.save+'.txt', options.save+'_vtree.txt')
            else:
                compile_sdd(options.input_dt, options.save+'.txt', options.save+'_vtree.txt')

            print(options.save+'_vtree.txt')

            compile_sdd_time_end = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                   resource.getrusage(resource.RUSAGE_SELF).ru_utime - compile_sdd_time_start
            print(f"compile {basename} DT to SDD in {compile_sdd_time_end:.1f} secs")
            # with open(f'results/compile_time_sdd.txt', 'a') as f:
            #     f.write(f"{data_name} & {compile_sdd_time_end:.1f}\n")

        elif options.train_classifier == 'ddnnf':
            compile_ddnnf_time_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                       resource.getrusage(resource.RUSAGE_SELF).ru_utime
            if options.ohe:
                rodt_to_ddnnf_ohe(options.dataset, options.input_dt, options.save+'.pkl')
            else:
                rodt_to_ddnnf(options.dataset, options.input_dt, options.save+'.pkl')
            compile_ddnnf_time_end = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                     resource.getrusage(resource.RUSAGE_SELF).ru_utime - compile_ddnnf_time_start
            print(f"compile {basename} DT to d-DNNF in {compile_ddnnf_time_end:.1f} secs")
            # with open(f'results/compile_time_ddnnf.txt', 'a') as f:
            #     f.write(f"{data_name} & {compile_ddnnf_time_end:.1f}\n")
    else:
        bench_name = options.reproduce

        with open(f"{bench_name}", 'r') as fp:
            name_list = fp.readlines()

        if not options.ohe:
            for item in name_list:
                data_name = item.strip()
                # dataset and dt
                data = f"datasets/{data_name}.csv"
                dt_save = f"models/dts/binary/{data_name}.pkl"
                # save d-DNNF files
                save_ddnnf = f"models/ddnnfs/"
                # sdd and vtree
                sdd_save = f"models/sdds/{data_name}.txt"
                vtree_save = f"models/sdds/{data_name}_vtree.txt"
                print(f"############ {data_name} ############")
                if options.train_classifier == 'sdd':
                    compile_sdd_time_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                             resource.getrusage(resource.RUSAGE_SELF).ru_utime
                    compile_sdd(dt_save, None, None)
                    compile_sdd_time_end = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                           resource.getrusage(resource.RUSAGE_SELF).ru_utime - compile_sdd_time_start
                    print(f"compile {data_name} to SDD in {compile_sdd_time_end:.1f} secs")
                    # with open(f'results/compile_time_sdd.txt', 'a') as f:
                    #     f.write(f"{data_name} & {compile_sdd_time_end:.1f}\n")
                elif options.train_classifier == 'ddnnf':
                    compile_ddnnf_time_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                               resource.getrusage(resource.RUSAGE_SELF).ru_utime
                    rodt_to_ddnnf(data, dt_save, None)
                    compile_ddnnf_time_end = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                             resource.getrusage(
                                                 resource.RUSAGE_SELF).ru_utime - compile_ddnnf_time_start
                    print(f"compile {data_name} to d-DNNF in {compile_ddnnf_time_end:.1f} secs")
                    # with open(f'results/compile_time_ddnnf.txt', 'a') as f:
                    #     f.write(f"{data_name} & {compile_ddnnf_time_end:.1f}\n")
        else:
            for item in name_list:
                # dataset and dt
                data_name = item.strip()
                # dataset and dt
                data_ohe = f"datasets/{data_name}.csv"
                dt_save_ohe = f"models/dts/categorical/{data_name}.pkl"
                # save d-DNNF files
                save_ddnnf = f"models/ddnnfs/"
                # sdd and vtree
                sdd_save_ohe = f"models/sdds/{data_name}.txt"
                vtree_save_ohe = f"models/sdds/{data_name}_vtree.txt"
                print(f"############ {data_name} ############")
                if options.train_classifier == 'sdd':
                    compile_sdd_time_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                             resource.getrusage(resource.RUSAGE_SELF).ru_utime
                    compile_sdd_ohe(dt_save_ohe, None, None)
                    compile_sdd_time_end = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                           resource.getrusage(
                                               resource.RUSAGE_SELF).ru_utime - compile_sdd_time_start
                    print(f"compile {data_name} to SDD in {compile_sdd_time_end:.1f} secs")
                    # with open(f'results/compile_time_sdd.txt', 'a') as f:
                    #     f.write(f"{data_name} & {compile_sdd_time_end:.1f}\n")
                elif options.train_classifier == 'ddnnf':
                    compile_ddnnf_time_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                               resource.getrusage(resource.RUSAGE_SELF).ru_utime
                    rodt_to_ddnnf_ohe(data_ohe, dt_save_ohe, None)
                    compile_ddnnf_time_end = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                             resource.getrusage(
                                                 resource.RUSAGE_SELF).ru_utime - compile_ddnnf_time_start
                    print(f"compile {data_name} to d-DNNF in {compile_ddnnf_time_end:.1f} secs")
                    # with open(f'results/compile_time_ddnnf.txt', 'a') as f:
                    #     f.write(f"{data_name} & {compile_ddnnf_time_end:.1f}\n")
