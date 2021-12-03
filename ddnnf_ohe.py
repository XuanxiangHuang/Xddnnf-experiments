#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Explaining d-DNNF, SDD Classifiers (one-hot-encoding)
#   Author: Xuanxiang Huang
#
################################################################################
import collections
import pickle
from graphviz import Source
from pysdd.sdd import SddManager, Vtree
from sklearn.metrics import accuracy_score
from decision_tree_ohe import DecisionTree
import numpy as np
################################################################################


class SDD(object):
    """
        SDD classifier manager
    """
    def __init__(self):
        self.root = None  # root node
        self.b_nv = None
        self.o_nv = None
        self.b_feats = None
        self.o_feats = None
        self.sup_b_vars = None
        self.sup_o_vars = None
        self.to_b_feats = None
        self.to_o_feats = None

    def compile(self, in_model, b_feats, o_feats, to_b_feats, to_o_feats,
                sdd_file=None, vtree_file=None):
        """
            Compiling models to SDD classifier.
            :param in_model: a Boolean network,
                            where each element is a If-Then-Else gate (var, low, high).
            :param o_feats: original features
            :param b_feats: binarized features
            :param to_o_feats: map from original features to binarized features
            :param to_b_feats: map from binarized features to original features
            :param sdd_file: save sdd to file
            :param vtree_file: save vtree to file
            :return: a SDD classifier.
        """
        self.b_nv = len(b_feats)
        self.o_nv = len(o_feats)
        self.b_feats = b_feats
        self.o_feats = o_feats
        self.to_b_feats = to_b_feats
        self.to_o_feats = to_o_feats
        vtree = Vtree(var_count=self.b_nv, vtree_type="balanced")
        sdd = SddManager.from_vtree(vtree)
        # enable auto minimization and GC
        sdd.auto_gc_and_minimize_on()
        # BNet => SDD
        self.compile_bnet_sdd(in_model, sdd)

        # do GC manually to clean unsupport variables
        sdd.garbage_collect()

        # get support variables
        self.sup_b_vars = [None] * self.b_nv
        all_vars = [_ for _ in sdd.vars]
        for i in range(self.b_nv):
            # literal start from 1, i-th literal == i-1-th variable
            lit = all_vars[i].literal
            assert (lit == i + 1)
            neglit = -all_vars[i].literal
            if sdd.is_var_used(lit) or sdd.is_var_used(neglit):
                self.sup_b_vars[i] = all_vars[i]
        # get original support variables
        self.sup_o_vars = [None] * self.o_nv
        for i in range(self.o_nv):
            tmp = []
            feat = self.o_feats[i]
            for j in range(self.b_nv):
                if self.to_o_feats[self.b_feats[j]] == feat and self.sup_b_vars[j]:
                    tmp.append(self.sup_b_vars[j])
            if tmp:
                self.sup_o_vars[i] = tmp

        if sdd_file is not None and vtree_file is not None:
            # save sdd and vtree into files
            sdd_model = bytes(sdd_file, 'utf-8')
            self.root.save(sdd_model)
            vtree_model = bytes(vtree_file, 'utf-8')
            vtree.save(vtree_model)

    def compile_bnet_sdd(self, bnet, sdd: SddManager):
        """
            Compiling Boolean network into an SDD.

            :param bnet: a list of Boolean network gates.
            :param sdd: SDD manager.
            :return: an SDD classifier consistent with ordered rules,
                        i.e. they predict same instance to same target classes.
        """
        vars = [_ for _ in sdd.vars]
        g_nums = [g_num for g_num in bnet]
        g_nums.sort()
        gate2sdd = dict()
        gate2sdd.update({0: sdd.false()})
        gate2sdd.update({1: sdd.true()})
        for g_num in g_nums:
            gate = bnet[g_num]
            f = vars[gate[0]]
            g = gate2sdd.get(gate[1])
            h = gate2sdd.get(gate[2])
            g.ref()
            h.ref()
            assert (g is not None and h is not None)
            nf = ~f
            nf.ref()
            out1 = f & g
            out1.ref()
            out2 = nf & h
            out2.ref()
            out = out1 | out2
            out.ref()
            out1.deref()
            out2.deref()
            nf.deref()
            g.deref()
            h.deref()
            gate2sdd.update({g_num: out})
        root = gate2sdd[g_nums[-1]]
        root.ref()
        self.root = root

    def total_assignment(self, inst):
        """
            Get terminal node given a total assignment.

            :param inst: a total assignment.
            :return: true if assignment evaluated to terminal one else false.
        """
        assert (len(inst) == self.b_nv)
        lits = [None] * self.b_nv

        for j in range(self.b_nv):
            if self.sup_b_vars[j]:
                if int(inst[j]):
                    lits[j] = self.sup_b_vars[j].literal
                else:
                    lits[j] = -self.sup_b_vars[j].literal

        out = self.root
        for item in lits:
            if item:
                out = out.condition(item)
        assert out.is_true() or out.is_false()
        return out.is_true()

    def predict_one(self, in_x):
        """
            Return prediction of one given instance.

            :param in_x: total instance (not partial instance).
            :return: prediction of this instance.
        """
        inst = [int(pt) for pt in in_x]
        return self.total_assignment(inst)

    def predict_all(self, in_x):
        """
            Return a list of prediction given a list of instances.

            :param in_x: a list of total instances.
            :return: predictions of all instances.
        """
        y_pred = []
        for ins in in_x:
            inst = [int(pt) for pt in ins]
            y_pred.append(self.total_assignment(inst))
        return y_pred

    def accuracy(self, in_x, y_true):
        """
            Compare the output of sdd and desired prediction
        :param in_x: a list of total instances.
        :param y_true: desired prediction
        :return: accuracy in float.
        """
        y_pred = []
        for ins in in_x:
            inst = [int(pt) for pt in ins]
            y_pred.append(self.total_assignment(inst))
        acc = accuracy_score(y_true, y_pred)
        return acc

################################################################################
################################################################################


class dDNNFnode(object):
    """
        dDNNF node: 5-tuple (var, low, high, n_id, n_type).
        var: literal string or empty,
                var is None if the node representing AND, OR, T or F.
                var is not None if the node literal.
        children: children nodes.
        n_id: node index.
        n_type: node type:
                'AND' for AND-node, 'OR' for OR-node, 'L' for LEAF-node.
    """

    def __init__(self, var=None, children=None, n_id=None, n_type=None):
        self.var = var
        self.children = children
        self.n_id = n_id
        self.n_type = n_type


class dDNNF(object):
    """
        d-DNNF classifier manager.
    """
    def __init__(self):
        self.nn = None          # num of nodes
        self.root = None        # root node
        self.b_nv = None
        self.o_nv = None
        self.b_feats = None
        self.o_feats = None
        self.sup_b_vars = None
        self.sup_o_vars = None
        self.to_b_feats = None
        self.to_o_feats = None

        self.F = None           # False leaf
        self.T = None           # True leaf
        # node index to node object.
        # 0 is unused, 1 for T, -1 for F.
        # for i >= 2, i for features[i-2]
        self.idx2nd = None
        self.pos_lits = None    # supported positive literals
        self.neg_lits = None    # supported negative literals
        # next node index to be allocated, due to reduction, n_idx != nn
        self.n_idx = None

    def compile(self, in_model, b_feats, o_feats, to_b_feats, to_o_feats):
        """
            Transform ROBDT(Read-Once Binary Decision Tree) to sd-DNNF.
            :param in_model: given ROBDT.
            :param o_feats: original features
            :param b_feats: binarized features
            :param to_o_feats: map from binarized features to original features
            :param to_b_feats: map from original features to binarized features
            :return: sd-DNNF model.
        """
        if type(in_model) != DecisionTree:
            print('unknown input model, support RODT')
            assert False

        self.b_nv = len(b_feats)
        self.o_nv = len(o_feats)
        self.b_feats = b_feats
        self.o_feats = o_feats
        self.to_b_feats = to_b_feats
        self.to_o_feats = to_o_feats

        if type(in_model) == DecisionTree:
            self.dt2dDNNF(in_model)

    def dt2dDNNF(self, dt):
        """
            Convert a RODT into d-DNNF (more specifically dec-DNNF).
            All feature domain must be binary, 0 for leaf 0, 1 for leaf 1, non-leaf start from 2.
            :param dt: given RODT.
            :return: a dictionary, from index of ITE-gate to (var, low, high) tuple.
                    Note that ITE-gate is a decision node of d-DNNF.
        """
        #####################################################
        def dt2Bnet(dt):
            nd2ite = dict()
            nd2idx = dict()
            idx = [2]
            __dt2Bnet(dt.classifier.root, nd2ite, nd2idx, idx)
            return nd2ite

        def __dt2Bnet(node, n2ite, n2idx, idx):
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

            __dt2Bnet(n0, n2ite, n2idx, idx)
            __dt2Bnet(n1, n2ite, n2idx, idx)

            idx_0 = n2idx[n0]
            idx_1 = n2idx[n1]

            assert n0.description == '≤ 0' and n1.description == '> 0'
            n2ite.update({idx[0]: (node.attr.name, idx_1, idx_0)})

            n2idx.update({node: idx[0]})
            idx[0] += 1

            return
        #####################################################
        ######################### Bnet to d-DNNF #########################
        dic_ddnnf = dt2Bnet(dt)
        pos_lits = []
        neg_lits = []
        self.F = dDNNFnode(n_id=-1, n_type='L')
        self.T = dDNNFnode(n_id=1, n_type='L')
        self.idx2nd = dict()
        self.idx2nd.update({-1: self.F})
        self.idx2nd.update({1: self.T})
        n_idx = 2
        for i in range(self.b_nv):
            pnd = dDNNFnode(var=self.b_feats[i], n_id=n_idx, n_type='L')
            nnd = dDNNFnode(var=self.b_feats[i], n_id=-n_idx, n_type='L')
            pos_lits.append(pnd)
            neg_lits.append(nnd)
            self.idx2nd.update({n_idx: pnd})
            self.idx2nd.update({-n_idx: nnd})
            n_idx += 1

        g2n = dict()
        g2n.update({0: self.F})
        g2n.update({1: self.T})
        for gate in dic_ddnnf:
            feat, low, high = dic_ddnnf[gate]
            f_id = self.b_feats.index(feat)
            leaf = pos_lits[f_id]
            nleaf = neg_lits[f_id]
            assert leaf.var == feat and nleaf.var == feat
            if low == 1 or g2n[low] == self.T:
                l_and = leaf
            elif low == 0 or g2n[low] == self.F:
                l_and = self.F
            else:
                # create a new AND node
                l_and = dDNNFnode(children=[leaf, g2n[low]], n_id=n_idx, n_type='AND')
                self.idx2nd.update({n_idx: l_and})
                n_idx += 1
            if high == 1 or g2n[high] == self.T:
                h_and = nleaf
            elif high == 0 or g2n[high] == self.F:
                h_and = self.F
            else:
                # create a new AND node
                h_and = dDNNFnode(children=[nleaf, g2n[high]], n_id=n_idx, n_type='AND')
                self.idx2nd.update({n_idx: h_and})
                n_idx += 1
            if l_and is self.F:
                g2n.update({gate: h_and})
            elif h_and is self.F:
                g2n.update({gate: l_and})
            elif l_and is self.T:
                g2n.update({gate: l_and})
            elif h_and is self.T:
                g2n.update({gate: h_and})
            else:
                # create a new OR-node
                or_node = dDNNFnode(children=[l_and, h_and], n_id=n_idx, n_type='OR')
                self.idx2nd.update({n_idx: or_node})
                g2n.update({gate: or_node})
                n_idx += 1
        self.root = self.idx2nd[n_idx - 1]
        self.n_idx = n_idx
        self.nn = self.size(self.root)
        ######################### Bnet to d-DNNF #########################
        ######################### Collect support features #########################
        assert len(pos_lits) == self.b_nv
        assert len(neg_lits) == self.b_nv
        collect_sup = set()
        for nd in self.dfs_postorder(self.root):
            if nd.n_type != 'L':
                continue
            assert nd.var, "no reference to TRUE and FALSE"
            collect_sup.add(nd.var)
        self.sup_b_vars = [None] * self.b_nv
        self.pos_lits = [None] * self.b_nv
        self.neg_lits = [None] * self.b_nv
        for i in range(self.b_nv):
            if self.b_feats[i] in collect_sup:
                self.sup_b_vars[i] = self.b_feats[i]
                self.pos_lits[i] = pos_lits[i]
                self.neg_lits[i] = neg_lits[i]
        for ii in range(self.b_nv):
            if self.sup_b_vars[ii]:
                assert self.sup_b_vars[ii] == self.pos_lits[ii].var
                assert self.sup_b_vars[ii] == self.neg_lits[ii].var
            else:
                assert self.pos_lits[ii] is None
                assert self.neg_lits[ii] is None

        ######################### Collect sup features #########################
        ######################### Check sd-DNNF #########################
        print("do smooth")
        self.smooth()
        # no reference to TRUE and FALSE
        for nd in self.dfs_postorder(self.root):
            if nd.n_type != 'L':
                continue
            assert nd.var, "no reference to TRUE and FALSE"
        ######################### Check sd-DNNF #########################

    def total_assignment(self, inst):
        """
            Get prediction node given a total assignment.
            :param inst: a total assignment.
            :return: true if assignment evaluated to terminal one else false.
        """
        assert (len(inst) == self.b_nv)
        feats = self.b_feats

        assign = dict()
        assign.update({self.F: 0})
        assign.update({self.T: 1})
        for nd in self.dfs_postorder(self.root):
            if nd.n_type == 'L':
                if nd.var:
                    idx = feats.index(nd.var)
                    if int(inst[idx]) and nd.n_id > 0:
                        assign.update({nd: 1})
                    elif not int(inst[idx]) and nd.n_id < 0:
                        assign.update({nd: 1})
                    else:
                        assign.update({nd: 0})
            else:
                tmp = [assign[chd] for chd in nd.children]
                if nd.n_type == 'AND':
                    if 0 in tmp:
                        assign.update({nd: 0})
                    else:
                        assign.update({nd: 1})
                else:
                    if 1 in tmp:
                        assign.update({nd: 1})
                    else:
                        assign.update({nd: 0})

        assert assign[self.root] == 1 or assign[self.root] == 0
        if assign[self.root] == 1:
            return 1
        else:
            return 0

    def predict_one(self, in_x):
        """
            Return prediction of one given instance.

            :param in_x: total instance (not partial instance).
            :return: prediction of this instance.
        """
        inst = [int(pt) for pt in in_x]
        return self.total_assignment(inst)

    def predict_all(self, in_x):
        """
            Return a list of prediction given a list of instances.

            :param in_x: a list of total instances.
            :return: predictions of all instances.
        """
        y_pred = []
        for ins in in_x:
            inst = [int(pt) for pt in ins]
            y_pred.append(self.total_assignment(inst))
        return y_pred

    def accuracy(self, in_x, y_true):
        """
            Compare the output of d-DNNF and desired prediction
        :param in_x: a list of total instances.
        :param y_true: desired prediction
        :return: accuracy in float.
        """
        y_pred = []
        for ins in in_x:
            inst = [int(pt) for pt in ins]
            y_pred.append(self.total_assignment(inst))
        acc = accuracy_score(y_true, y_pred)
        return acc

    def bfs(self, root):
        """
            Iterate through nodes in breadth first search (BFS) order.

            :param root: root node of decision tree.
            :return: a set of all tree nodes in BFS order.
        """

        #####################################################
        def _bfs(node, visited):
            queue = collections.deque()
            queue.appendleft(node)
            while queue:
                nd = queue.pop()
                if nd not in visited:
                    if nd.children:
                        for chd in nd.children:
                            queue.appendleft(chd)
                    visited.add(nd)
                    yield nd

        #####################################################
        yield from _bfs(root, set())

    def dfs_postorder(self, root):
        """
            Iterate through nodes in depth first search (DFS) post-order.

            :param root: root node of d-DNNF.
            :return: a set of nodes in DFS-post-order.
        """

        #####################################################
        def _dfs_postorder(nd, visited):
            if nd.children:
                for chd in nd.children:
                    yield from _dfs_postorder(chd, visited)
            if nd not in visited:
                visited.add(nd)
                yield nd

        #####################################################
        yield from _dfs_postorder(root, set())

    def size(self, root):
        """
            Return size of d-DNNF rooted at given node.

            :param root: root node.
            :return: size of d-DNNF.
        """

        #####################################################
        def _size(nd, visited):
            if nd.n_id in visited:
                return
            if nd.n_id not in visited:
                visited.add(nd.n_id)
            if nd.children:
                for chd in nd.children:
                    _size(chd, visited)

        #####################################################
        counter = set()
        _size(root, counter)
        return len(counter)

    def to_dot(self, name='d_DNNF'):
        """
            Convert to DOT language representation.
            See the
            `DOT language reference <http://www.graphviz.org/content/dot-language>`_
            for details.
        """
        parts = ['graph', name, '{']
        for nd in self.dfs_postorder(self.root):
            if nd is self.F:
                parts += ['n' + str(nd.n_id).replace('-', '_'), '[label=0,shape=box];']
            elif nd is self.T:
                parts += ['n' + str(nd.n_id), '[label=1,shape=box];']
            elif nd.n_type == 'L':
                feat_name = nd.var
                if nd.n_id < 0:
                    feat_name = '- ' + feat_name
                parts.append('n' + str(nd.n_id).replace('-', '_'))
                parts.append('[label="{0}",shape=box];'.format(feat_name))
            else:
                assert nd.n_id > 0
                parts.append('n' + str(nd.n_id))
                if nd.n_type == 'AND':
                    parts.append('[label="#{0}",shape=triangle];'.format(nd.n_id))
                else:
                    parts.append('[label="#{0}",shape=invtriangle];'.format(nd.n_id))
        for nd in self.dfs_postorder(self.root):
            if nd.children:
                assert nd.n_id > 0
                for chd in nd.children:
                    parts += ['n' + str(nd.n_id), '--',
                              'n' + str(chd.n_id).replace('-', '_'),
                              '[label=""];']
        parts.append('}')
        return " ".join(parts)

    def dump_figure(self, filename):
        """
            Dump d-DNNF into figure, currently in .pdf format.
            :param filename: filename without .pdf
            :return: a pdf file sotring d-DNNF strucure.
        """
        ddnnf_dot = self.to_dot()
        file = open(filename + '.gv', 'w')
        file.write(ddnnf_dot)
        file.close()
        ddnnf_graph = Source.from_file(filename=filename + '.gv')
        ddnnf_graph.render(filename=filename + '.gv')

    @staticmethod
    def save_model(ddnnf, filename):
        """
            Save d-DNNF to pickle model.

            :param ddnnf: given d-DNNF classifier.
            :param filename: file storing d-DNNF.
            :return: none.
        """
        with open(filename, "wb") as f:
            pickle.dump(ddnnf, f)

    @classmethod
    def from_file(cls, filename):
        """
            Load d-DNNF classifier from file.

            :param filename: d-DNNF in pickle.
            :return: d-DNNF classifier.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def smooth(self):
        """
            Make d-DNNF smooth (sd-DNNF)
        """
        lookup = dict()
        for nd in self.dfs_postorder(self.root):
            if nd.n_type != 'OR':
                continue
            assert nd.children, "non-leaf node has children"
            # do smooth for OR-nodes
            chds_vars = []
            all_vars = set()
            for chd in nd.children:
                chd_var = set()
                for leaf in self.dfs_postorder(chd):
                    if leaf.n_type != 'L':
                        continue
                    elif leaf is self.T:
                        assert False, "encounter TRUE"
                    elif leaf is self.F:
                        assert False, "encounter FALSE"
                    chd_var.add(leaf.var)
                    all_vars.add(leaf.var)
                assert chd_var, "non-empty"
                chds_vars.append(chd_var)
            # check if vars of child_i and vars of root node are the same
            for i in range(len(nd.children)):
                assert chds_vars[i].issubset(all_vars)
                if chds_vars[i] != all_vars:
                    tmp = all_vars.difference(chds_vars[i])
                    new_and_chd = [nd.children[i]]
                    for feat in tmp:
                        f_id = self.sup_b_vars.index(feat)
                        leaf = self.pos_lits[f_id]
                        nleaf = self.neg_lits[f_id]
                        assert leaf.var == feat and nleaf.var == feat
                        key = (leaf, nleaf, 'OR')
                        if lookup.get(key) is not None:
                            new_or_nd = lookup[key]
                        else:
                            new_or_nd = dDNNFnode(children=[leaf, nleaf], n_id=self.n_idx, n_type='OR')
                            lookup.update({(leaf, nleaf, 'OR'): new_or_nd})
                            self.n_idx += 1
                        new_and_chd.append(new_or_nd)
                    new_and_nd = dDNNFnode(children=new_and_chd, n_id=self.n_idx, n_type='AND')
                    self.n_idx += 1
                    nd.children[i] = new_and_nd

        # check if resulting d-DNNF is sd-DNNF
        for nd in self.dfs_postorder(self.root):
            if nd.n_type != 'OR':
                continue
            chds_vars = []
            all_vars = set()
            for chd in nd.children:
                chd_var = set()
                for leaf in self.dfs_postorder(chd):
                    if leaf.n_type != 'L':
                        continue
                    elif leaf is self.T:
                        assert False, "encounter TRUE"
                    elif leaf is self.F:
                        assert False, "encounter FALSE"
                    chd_var.add(leaf.var)
                    all_vars.add(leaf.var)
                assert chd_var, "non-empty"
                chds_vars.append(chd_var)
            # check if vars of child_i and vars of root node are the same
            for i in range(len(nd.children)):
                assert chds_vars[i] == all_vars

    def check_ICO_VA(self, inst, va=True):
        """
            Given a list of value (partial instance), check validity or inconsistency
            return True if it is valid (resp. inconsistent),
            else return False.
            :param inst: an instance
            :param va: True if check VA else check ICO
        """
        assert len(inst) == self.b_nv
        n_univ_var = 0
        assign = dict()
        for i in range(self.b_nv):
            leaf = self.pos_lits[i]
            nleaf = self.neg_lits[i]

            if self.sup_b_vars[i]:
                assert leaf.var == self.sup_b_vars[i] and nleaf.var == self.sup_b_vars[i]
            else:
                assert leaf is None and nleaf is None
                continue

            if inst[i] is None:
                assign.update({leaf: 1})
                assign.update({nleaf: 1})
                n_univ_var += 1
            elif int(inst[i]) == 1:
                assign.update({leaf: 1})
                assign.update({nleaf: 0})
            elif int(inst[i]) == 0:
                assign.update({leaf: 0})
                assign.update({nleaf: 1})
            else:
                assert False

        for nd in self.dfs_postorder(self.root):
            if nd.n_type == 'L':
                continue
            if nd.n_type == 'AND':
                num = 1
                for chd in nd.children:
                    num *= assign[chd]
                assign.update({nd: num})
            elif nd.n_type == 'OR':
                num = 0
                for chd in nd.children:
                    num += assign[chd]
                assign.update({nd: num})
        n_model = assign[self.root]
        assert n_univ_var >= 0

        if va:
            return n_model == 2 ** n_univ_var
        else:
            return n_model == 0
