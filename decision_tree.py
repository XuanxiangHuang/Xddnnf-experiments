#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Decision Tree Classifier
#   Author: Xuanxiang Huang
#
################################################################################
from Orange.classification import TreeLearner
from Orange.data import Table
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
        self.datatable = Table(dataset)
        self.features = [att.name for att in self.datatable.domain.attributes]
        self.classes = self.datatable.domain.class_var.values

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.datatable.X, self.datatable.Y, test_size=learning_rate)

        self.learner = TreeLearner(max_depth=max_depth, binarize=True)
        self.classifier = None

    def train(self):
        """
            Training decision tree with given datasets.

            :return: none.
        """
        train_datatable = Table.from_numpy(self.datatable.domain, self.X_train, self.y_train)
        self.classifier = self.learner(train_datatable)

        train_acc = accuracy_score(self.y_train, self.classifier(self.X_train))
        test_acc = accuracy_score(self.y_test, self.classifier(self.X_test))

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

    def bdt2bnet(self):
        """
            Convert each Binary DT node (var, low, high) to an ITE(F, G, H).
            All feature domain must be binary, 0 for leaf 0, 1 for leaf 1, non-leaf start from 2.

            :return: a dictionary, from index of ITE-gate to (var, low, high) tuple.
        """

        ############################################
        def __bdt2bnet(node, n2ite, n2idx, idx):
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

            __bdt2bnet(n0, n2ite, n2idx, idx)
            __bdt2bnet(n1, n2ite, n2idx, idx)

            idx_0 = n2idx[n0]
            idx_1 = n2idx[n1]

            if int(n0.description):
                # associate each ITE gate with an unique idx
                n2ite.update({idx[0]: (node.attr_idx, idx_0, idx_1)})
            else:
                n2ite.update({idx[0]: (node.attr_idx, idx_1, idx_0)})

            n2idx.update({node: idx[0]})
            idx[0] += 1

            return

        ############################################

        for item in self.datatable.domain.attributes:
            if len(item.values) != 2:
                print('Feature domain is not binary')
                assert False
            v0 = item.values[0]
            v1 = item.values[1]
            if (v0 == v1) or (v0 not in ('0', '1')) or (v1 not in ('0', '1')):
                print('Feature domain is not binary')
                assert False
        nd2ite = dict()
        nd2idx = dict()
        idx = [2]
        __bdt2bnet(self.classifier.root, nd2ite, nd2idx, idx)
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
