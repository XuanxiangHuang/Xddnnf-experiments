#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Decision Tree Classifier (one-hot encoding)
#   Author: Xuanxiang Huang
#
################################################################################
from Orange.classification import TreeLearner
from Orange.data import Table
from Orange.preprocess import Continuize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
################################################################################


class DecisionTree:
    """
        Use Orange, learning DT classifier from data,
        features could be: binary, categorical or continuous.
    """

    def __init__(self, dataset, max_depth=7, learning_rate=0.2):
        self.train_datatable = None
        self.test_datatable = None
        self.train_datatable_bin = None
        self.test_datatable_bin = None
        self.features_bin = None
        self.classes_bin = None
        self.continuizer = None
        self.feat_to_original = dict()
        self.feat_to_binarize = dict()

        # original features and classes
        self.datatable = Table(dataset)
        self.features = [att.name for att in self.datatable.domain.attributes]
        self.classes = self.datatable.domain.class_var.values

        # split dataset
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.datatable.X, self.datatable.Y, test_size=learning_rate)

        # learner and classifier
        self.learner = TreeLearner(max_depth=max_depth, binarize=True)
        self.classifier = None

    def train(self):
        """
            Training decision tree with given datasets.
            Apply one-hot-encoding to binarize CAT features.

            :return: none.
        """
        self.train_datatable = Table.from_numpy(self.datatable.domain, self.X_train, self.y_train)
        self.test_datatable = Table.from_numpy(self.datatable.domain, self.X_test, self.y_test)

        # binarized features and classes
        self.continuizer = Continuize()
        self.train_datatable_bin = self.continuizer(self.train_datatable)
        self.test_datatable_bin = self.continuizer(self.test_datatable)

        assert len(self.train_datatable_bin) == len(self.train_datatable)
        assert len(self.test_datatable_bin) == len(self.test_datatable)

        self.features_bin = [att.name for att in self.train_datatable_bin.domain.attributes]
        self.classes_bin = self.train_datatable_bin.domain.class_var.values

        # feature mapping (to_original, and to_binarize)
        for item in self.train_datatable_bin.domain.attributes:
            self.feat_to_original.update({item.name: item.compute_value.variable.name})

        for original_f in self.datatable.domain.attributes:
            tmp = []
            for binarize_f in self.train_datatable_bin.domain.attributes:
                if self.feat_to_original[binarize_f.name] == original_f.name:
                    tmp.append(binarize_f.name)
            self.feat_to_binarize.update({original_f.name: tmp})

        self.classifier = self.learner(self.train_datatable_bin)

        train_acc = accuracy_score(self.train_datatable_bin.Y, self.classifier(self.train_datatable_bin.X))
        test_acc = accuracy_score(self.test_datatable_bin.Y, self.classifier(self.test_datatable_bin.X))

        return round(train_acc, 3), round(test_acc, 3)

    def __str__(self):
        res = ''
        res += ('#Total Node: {0}\n'.format(self.classifier.node_count()))
        res += ('#Leaf Node: {0}\n'.format(self.classifier.leaf_count()))
        res += ('Depth: {0}\n'.format(self.classifier.depth()))
        res += self.classifier.print_tree()
        return res

    def size(self):
        return self.classifier.node_count()

    @staticmethod
    def save_model(dt, filename):
        """
            Save DT to pickle model.

            :param dt: decision tree classifier.
            :param filename: filename storing dt classifier.
            :return: none.
        """
        with open(filename, "wb") as f:
            pickle.dump(dt, f)

    @classmethod
    def from_file(cls, filename):
        """
            Load DT classifier from file.

            :param filename: decision tree classifier in pickle.
            :return: decision tree.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def predict_one(self, in_x):
        """
            Get prediction of given one instance or sample.

            :param in_x: given instance.
            :return: prediction.
        """
        return self.classifier(in_x)

    def dt2bnet(self):
        """
            Convert each Binarized DT node (var, low, high) to an ITE(F, G, H).
            All feature domain must be binary, 0 for leaf 0, 1 for leaf 1, non-leaf start from 2.

            :return: a dictionary, from index of ITE-gate to (var, low, high) tuple.
        """

        ############################################
        def __dt2bnet(node, n2ite, n2idx, idx):
            if not node.children:
                probs = node.value / np.sum(node.value)
                target = np.argmax(probs, axis=-1)
                if int(target):
                    n2idx.update({node: 1})
                else:
                    n2idx.update({node: 0})
                return

            n0 = node.children[0]
            n1 = node.children[1]

            __dt2bnet(n0, n2ite, n2idx, idx)
            __dt2bnet(n1, n2ite, n2idx, idx)

            idx_0 = n2idx[n0]
            idx_1 = n2idx[n1]

            assert n0.description == '≤ 0' and n1.description == '> 0'
            assert self.features_bin[node.attr_idx] == node.attr.name
            n2ite.update({idx[0]: (node.attr_idx, idx_1, idx_0)})

            n2idx.update({node: idx[0]})
            idx[0] += 1

            return

        ############################################

        nd2ite = dict()
        nd2idx = dict()
        idx = [2]
        __dt2bnet(self.classifier.root, nd2ite, nd2idx, idx)
        return nd2ite

    def check_read_once(self):
        """
            Check if DT is read-once.

        :return: True or False.
        """
        ############### dfs traversal ###############
        def traverse(node, r, paths):
            for child_idx, child in enumerate(node.children):
                r.append(node.attr_idx)
                traverse(child, r, paths)
                r.pop()

        ############### get all paths ###############
        def get_paths(node):
            paths = []
            traverse(node, [], paths)
            return paths

        ############################################

        paths = get_paths(self.classifier.root)
        for ele in paths:
            tmp = set(ele)
            if len(ele) != len(tmp):
                print(f"DT is not read-once: path {ele} contain duplicated features")
                return False
        print(f"DT is read-once")
        return True