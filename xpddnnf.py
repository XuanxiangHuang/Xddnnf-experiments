#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   d-DNNF Classifiers explainer
#   Author: Xuanxiang Huang
#
################################################################################
import resource
from pysat.formula import IDPool
from pysat.solvers import Solver
from ddnnf import dDNNF
from ddnnf_ohe import dDNNF as dDNNF_OHE
################################################################################


class XpdDnnf(object):
    """
        Explain d-DNNF classifier.
    """
    def __init__(self, dDNNFman: dDNNF, verb=0):
        self.dDNNFman = dDNNFman
        self.nv = dDNNFman.nv
        assert self.nv == len(dDNNFman.sup_feats), "fatal error"
        self.verbose = verb

    def reachable(self, inst, univ, pred):
        """
            Check if desired prediction/class is reachable.

            :param inst: given an instance
            :param univ: list of universal features
            :param pred: desired prediction
            :return: True if reachable else False
        """
        assert len(inst) == self.nv
        inst_ = inst.copy()
        for i in range(self.nv):
            if univ[i]:
                inst_[i] = None

        if pred:
            return not self.dDNNFman.check_ICO_VA(inst_, va=False)
        else:
            return not self.dDNNFman.check_ICO_VA(inst_, va=True)

    def find_axp(self, inst, pred, fixed=None):
        """
            Compute one abductive explanation (Axp).

            :param fixed: a list of features declared as fixed.
            :param pred: prediction
            :param inst: given list of literals.
            :return: one abductive explanation,
                        each element in the return Axp is a feature index.
        """
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime

        assert len(inst) == self.nv
        # get/create fix array
        if not fixed:
            fix = [True] * self.nv
        else:
            fix = fixed.copy()
        assert (len(fix) == self.nv)
        # modify literals according to fix array
        inst_ = inst.copy()

        for i in range(self.nv):
            if not fix[i]:
                inst_[i] = None

        for i in range(self.nv):
            if fix[i]:
                fix[i] = not fix[i]
                inst_[i] = None

                if (pred and not self.dDNNFman.check_ICO_VA(inst_, va=True)) or \
                        (not pred and not self.dDNNFman.check_ICO_VA(inst_, va=False)):
                    inst_[i] = inst[i]
                    fix[i] = not fix[i]
        # axp is a subset of fixed features, and it is minimal
        axp = [i for i in range(self.nv) if fix[i]]

        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime - time

        if self.verbose:
            if self.verbose == 1:
                print(f"Axp: {axp}")
            elif self.verbose == 2:
                print(f"Axp: {axp} ({[self.dDNNFman.sup_feats[i] for i in axp]})")
            print("Runtime: {0:.3f}".format(time))

        return axp

    def find_cxp(self, inst, pred, universal=None):
        """
            Compute one contrastive explanation (Cxp).

            :param universal: a list of features declared as universal.
            :param inst: given an instance.
            :param pred: prediction
            :return: one contrastive explanation,
                        each element in the return Cxp is a feature index.
        """
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime

        assert len(inst) == self.nv
        # get/create univ array
        if not universal:
            univ = [True] * self.nv
        else:
            univ = universal.copy()
        assert (len(univ) == self.nv)
        # modify literals according to univ array
        inst_ = inst.copy()

        for i in range(self.nv):
            if univ[i]:
                inst_[i] = None

        for i in range(self.nv):
            if univ[i]:
                univ[i] = not univ[i]
                inst_[i] = inst[i]

                if (pred and self.dDNNFman.check_ICO_VA(inst_, va=True)) or \
                        (not pred and self.dDNNFman.check_ICO_VA(inst_, va=False)):
                    inst_[i] = None
                    univ[i] = not univ[i]

        # cxp is a subset of universal features, and it is minimal
        cxp = [i for i in range(self.nv) if univ[i]]

        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime - time

        if self.verbose:
            if self.verbose == 1:
                print(f"Cxp: {cxp}")
            elif self.verbose == 2:
                print(f"Cxp: {cxp} ({[self.dDNNFman.sup_feats[i] for i in cxp]})")
            print("Runtime: {0:.3f}".format(time))

        return cxp

    def enum_exps(self, inst, pred):
        """
            Enumerate all (abductive and contrastive) explanations, using MARCO algorithm.

            :param inst: given list of literals
            :param pred: prediction
            :return: a list of all Axps, a list of all Cxps.
        """

        #########################################
        vpool = IDPool()

        def new_var(name):
            """
                Inner function,
                Find or new a PySAT variable.
                See PySat.

                :param name: name of variable
                :return: index of variable
            """
            return vpool.id(f'{name}')

        #########################################

        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime

        assert len(inst) == self.nv

        num_axps = 0
        num_cxps = 0
        axps = []
        cxps = []
        slv = Solver(name="glucose3")
        for i in range(self.nv):
            new_var('u_{0}'.format(i))
        # initially all features are fixed
        univ = [False] * self.nv

        while slv.solve():
            # first model is empty
            model = slv.get_model()
            for lit in model:
                name = vpool.obj(abs(lit)).split(sep='_')
                univ[int(name[1])] = False if lit < 0 else True
            if self.reachable(inst, univ, not pred):
                cxp = self.find_cxp(inst, pred, univ)
                slv.add_clause([-new_var('u_{0}'.format(i))
                                for i in cxp])
                num_cxps += 1
                cxps.append(cxp)
            else:
                fix = [not i for i in univ]
                axp = self.find_axp(inst, pred, fix)
                assert self.check_one_axp(inst,pred,axp), "not an AXp"
                slv.add_clause([new_var('u_{0}'.format(i))
                                for i in axp])
                num_axps += 1
                axps.append(axp)

        slv.delete()

        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime - time

        print('#AXp:', num_axps)
        print('#CXp:', num_cxps)
        print("Runtime: {0:.3f}".format(time))

        return axps, cxps

    def check_one_axp(self, inst, pred, axp):
        """
            Check if given axp is 1) a weak AXp and 2) subset-minimal.

            :param inst: given instance
            :param pred: prediction
            :param axp: one abductive explanation (index).
            :return: true if given axp is an AXp
                        else false.
        """
        assert len(inst) == self.dDNNFman.nv
        fix = [False] * self.dDNNFman.nv
        for i in axp:
            fix[i] = True
        # modify literals according to fix array
        inst_ = inst.copy()
        for i in range(self.dDNNFman.nv):
            if not fix[i]:
                inst_[i] = None
        # 1) axp is a weak AXp
        if (pred and not self.dDNNFman.check_ICO_VA(inst_, va=True)) or \
                (not pred and not self.dDNNFman.check_ICO_VA(inst_, va=False)):
            print(f'given axp {axp} is not a weak AXp')
            return False
        # 2) axp is subset-minimal
        for i in range(self.dDNNFman.nv):
            if fix[i]:
                fix[i] = not fix[i]
                inst_[i] = None
                if (pred and not self.dDNNFman.check_ICO_VA(inst_, va=True)) or \
                        (not pred and not self.dDNNFman.check_ICO_VA(inst_, va=False)):
                    inst_[i] = inst[i]
                    fix[i] = not fix[i]
                else:
                    print(f'given axp {axp} is not subset-minimal')
                    return False
        return True


class XpdDnnf_ohe(object):
    """
        Explain d-DNNF classifier.
    """
    def __init__(self, dDNNFman: dDNNF_OHE, verb=0):
        self.dDNNFman = dDNNFman
        self.verbose = verb

    def reachable(self, inst, univ, pred):
        """
            Check if desired prediction/class is reachable.

            :param inst: given an instance
            :param univ: list of universal features
            :param pred: desired prediction
            :return: True if reachable else False
        """
        assert len(inst) == self.dDNNFman.b_nv
        assert len(univ) == self.dDNNFman.o_nv
        inst_ = inst.copy()
        for i in range(self.dDNNFman.o_nv):
            if univ[i]:
                original_feat = self.dDNNFman.o_feats[i]
                bin_feats = self.dDNNFman.to_b_feats[original_feat]
                for bf in bin_feats:
                    index = self.dDNNFman.b_feats.index(bf)
                    inst_[index] = None
        if pred:
            return not self.dDNNFman.check_ICO_VA(inst_, va=False)
        else:
            return not self.dDNNFman.check_ICO_VA(inst_, va=True)

    def find_axp(self, inst, pred, fixed=None):
        """
            Compute one abductive explanation (Axp).

            :param fixed: a list of features declared as fixed.
            :param pred: prediction
            :param inst: given list of literals.
            :return: one abductive explanation,
                        each element in the return Axp is a feature index.
        """
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime

        assert len(inst) == self.dDNNFman.b_nv
        # get/create fix array
        if not fixed:
            fix = [True] * self.dDNNFman.o_nv
        else:
            fix = fixed.copy()
        assert (len(fix) == self.dDNNFman.o_nv)
        # modify literals according to fix array
        inst_ = inst.copy()
        for i in range(self.dDNNFman.o_nv):
            if not fix[i]:
                original_feat = self.dDNNFman.o_feats[i]
                bin_feats = self.dDNNFman.to_b_feats[original_feat]
                for bf in bin_feats:
                    index = self.dDNNFman.b_feats.index(bf)
                    inst_[index] = None
        for i in range(self.dDNNFman.o_nv):
            if fix[i]:
                fix[i] = not fix[i]
                original_feat = self.dDNNFman.o_feats[i]
                bin_feats = self.dDNNFman.to_b_feats[original_feat]
                for bf in bin_feats:
                    index = self.dDNNFman.b_feats.index(bf)
                    inst_[index] = None
                if (pred and not self.dDNNFman.check_ICO_VA(inst_, va=True)) or \
                        (not pred and not self.dDNNFman.check_ICO_VA(inst_, va=False)):
                    fix[i] = not fix[i]
                    original_feat = self.dDNNFman.o_feats[i]
                    bin_feats = self.dDNNFman.to_b_feats[original_feat]
                    for bf in bin_feats:
                        index = self.dDNNFman.b_feats.index(bf)
                        inst_[index] = inst[index]
        # axp is a subset of fixed features, and it is minimal
        axp = [i for i in range(self.dDNNFman.o_nv) if fix[i]]

        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime - time

        if self.verbose:
            if self.verbose == 1:
                print(f"Axp: {axp}")
            elif self.verbose == 2:
                print(f"Axp: {axp} ({[self.dDNNFman.o_feats[i] for i in axp]})")
            print("Runtime: {0:.3f}".format(time))

        return axp

    def find_cxp(self, inst, pred, universal=None):
        """
            Compute one contrastive explanation (Cxp) using Canonicity and ConDitioning.

            :param universal: a list of features declared as universal.
            :param inst: given an instance.
            :param pred: prediction
            :return: one contrastive explanation,
                        each element in the return Cxp is a feature index.
        """
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime

        assert len(inst) == self.dDNNFman.b_nv
        # get/create univ array
        if not universal:
            univ = [True] * self.dDNNFman.o_nv
        else:
            univ = universal.copy()
        assert (len(univ) == self.dDNNFman.o_nv)
        # modify literals according to univ array
        inst_ = inst.copy()
        for i in range(self.dDNNFman.o_nv):
            if univ[i]:
                original_feat = self.dDNNFman.o_feats[i]
                bin_feats = self.dDNNFman.to_b_feats[original_feat]
                for bf in bin_feats:
                    index = self.dDNNFman.b_feats.index(bf)
                    inst_[index] = None
        for i in range(self.dDNNFman.o_nv):
            if univ[i]:
                univ[i] = not univ[i]
                original_feat = self.dDNNFman.o_feats[i]
                bin_feats = self.dDNNFman.to_b_feats[original_feat]
                for bf in bin_feats:
                    index = self.dDNNFman.b_feats.index(bf)
                    inst_[index] = inst[index]
                if (pred and self.dDNNFman.check_ICO_VA(inst_, va=True)) or \
                        (not pred and self.dDNNFman.check_ICO_VA(inst_, va=False)):
                    univ[i] = not univ[i]
                    original_feat = self.dDNNFman.o_feats[i]
                    bin_feats = self.dDNNFman.to_b_feats[original_feat]
                    for bf in bin_feats:
                        index = self.dDNNFman.b_feats.index(bf)
                        inst_[index] = None
        # cxp is a subset of universal features, and it is minimal
        cxp = [i for i in range(self.dDNNFman.o_nv) if univ[i]]

        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime - time

        if self.verbose:
            if self.verbose == 1:
                print(f"Cxp: {cxp}")
            elif self.verbose == 2:
                print(f"Cxp: {cxp} ({[self.dDNNFman.o_feats[i] for i in cxp]})")
            print("Runtime: {0:.3f}".format(time))

        return cxp

    def enum_exps(self, inst, pred):
        """
            Enumerate all (abductive and contrastive) explanations, using MARCO algorithm.

            :param inst: given list of literals
            :param pred: prediction
            :return: a list of all Axps, a list of all Cxps.
        """

        #########################################
        vpool = IDPool()

        def new_var(name):
            """
                Inner function,
                Find or new a PySAT variable.
                See PySat.

                :param name: name of variable
                :return: index of variable
            """
            return vpool.id(f'{name}')

        #########################################

        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime

        assert len(inst) == self.dDNNFman.b_nv

        num_axps = 0
        num_cxps = 0
        axps = []
        cxps = []
        slv = Solver(name="glucose3")
        for i in range(self.dDNNFman.o_nv):
            new_var('u_{0}'.format(i))
        # initially all features are fixed
        univ = [False] * self.dDNNFman.o_nv

        while slv.solve():
            # first model is empty
            model = slv.get_model()
            for lit in model:
                name = vpool.obj(abs(lit)).split(sep='_')
                univ[int(name[1])] = False if lit < 0 else True
            if self.reachable(inst, univ, not pred):
                cxp = self.find_cxp(inst, pred, univ)
                slv.add_clause([-new_var('u_{0}'.format(i))
                                for i in cxp])
                num_cxps += 1
                cxps.append(cxp)
            else:
                fix = [not i for i in univ]
                axp = self.find_axp(inst, pred, fix)
                slv.add_clause([new_var('u_{0}'.format(i))
                                for i in axp])
                num_axps += 1
                axps.append(axp)

        slv.delete()

        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime - time

        print('#AXp:', num_axps)
        print('#CXp:', num_cxps)
        print("Runtime: {0:.3f}".format(time))

        return axps, cxps

    ########################################################################
    def check_one_axp(self, inst, pred, axp):
        """
            Check if given axp is 1) a weak AXp and 2) subset-minimal.

            :param inst: given instance
            :param pred: prediction
            :param axp: one abductive explanation (index).
            :return: true if given axp is an AXp
                        else false.
        """
        assert len(inst) == self.dDNNFman.b_nv
        fix = [False] * self.dDNNFman.o_nv
        for i in axp:
            fix[i] = True
        # modify instance according to fix array
        inst_ = inst.copy()
        for i in range(self.dDNNFman.o_nv):
            if not fix[i]:
                original_feat = self.dDNNFman.o_feats[i]
                bin_feats = self.dDNNFman.to_b_feats[original_feat]
                for bf in bin_feats:
                    index = self.dDNNFman.b_feats.index(bf)
                    inst_[index] = None
        # 1) axp is a weak AXp
        if (pred and not self.dDNNFman.check_ICO_VA(inst_, va=True)) or \
                (not pred and not self.dDNNFman.check_ICO_VA(inst_, va=False)):
            print(f'given axp {axp} is not a weak AXp')
            return False
        # 2) axp is subset-minimal
        for i in range(self.dDNNFman.o_nv):
            if fix[i]:
                fix[i] = not fix[i]
                original_feat = self.dDNNFman.o_feats[i]
                bin_feats = self.dDNNFman.to_b_feats[original_feat]
                for bf in bin_feats:
                    index = self.dDNNFman.b_feats.index(bf)
                    inst_[index] = None
                if (pred and not self.dDNNFman.check_ICO_VA(inst_, va=True)) or \
                        (not pred and not self.dDNNFman.check_ICO_VA(inst_, va=False)):
                    fix[i] = not fix[i]
                    original_feat = self.dDNNFman.o_feats[i]
                    bin_feats = self.dDNNFman.to_b_feats[original_feat]
                    for bf in bin_feats:
                        index = self.dDNNFman.b_feats.index(bf)
                        inst_[index] = inst[index]
                else:
                    print(f'given axp {axp} is not subset-minimal')
                    return False
        return True

    def check_one_cxp(self, inst, pred, cxp):
        """
            Check if given cxp is 1) a weak CXp and 2) subset-minimal.

            :param inst: given instance
            :param pred: prediction
            :param cxp: one contrastive explanation (index).
            :return: true if given cxp is an CXp
                        else false.
        """
        assert len(inst) == self.dDNNFman.b_nv
        univ = [False] * self.dDNNFman.o_nv
        for i in cxp:
            univ[i] = True
        # modify instance according to univ array
        inst_ = inst.copy()
        for i in range(self.dDNNFman.o_nv):
            if univ[i]:
                original_feat = self.dDNNFman.o_feats[i]
                bin_feats = self.dDNNFman.to_b_feats[original_feat]
                for bf in bin_feats:
                    index = self.dDNNFman.b_feats.index(bf)
                    inst_[index] = None
        # 1) cxp is a weak CXp
        if (pred and self.dDNNFman.check_ICO_VA(inst_, va=True)) or \
                (not pred and self.dDNNFman.check_ICO_VA(inst_, va=False)):
            print(f'given cxp {cxp} is not a weak CXp')
            return False
        # 2) cxp is subset-minimal
        for i in range(self.dDNNFman.o_nv):
            if univ[i]:
                univ[i] = not univ[i]
                original_feat = self.dDNNFman.o_feats[i]
                bin_feats = self.dDNNFman.to_b_feats[original_feat]
                for bf in bin_feats:
                    index = self.dDNNFman.b_feats.index(bf)
                    inst_[index] = inst[index]
                if (pred and self.dDNNFman.check_ICO_VA(inst_, va=True)) or \
                        (not pred and self.dDNNFman.check_ICO_VA(inst_, va=False)):
                    univ[i] = not univ[i]
                    original_feat = self.dDNNFman.o_feats[i]
                    bin_feats = self.dDNNFman.to_b_feats[original_feat]
                    for bf in bin_feats:
                        index = self.dDNNFman.b_feats.index(bf)
                        inst_[index] = None
                else:
                    print(f'given cxp {cxp} is not subset-minimal')
                    return False
        return True

    ########################################################################

