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
                assert check_one_axp(self.dDNNFman,inst,pred,axp), "not an AXp"
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
################################################################################


def check_one_axp(dDNNFman: dDNNF, inst, pred, axp):
    """
        Check if given axp is 1) a weak AXp and 2) subset-minimal.

        :param inst: given instance
        :param pred: prediction
        :param axp: one abductive explanation (index).
        :return: true if given axp is an AXp
                    else false.
    """
    assert len(inst) == dDNNFman.nv
    fix = [False] * dDNNFman.nv
    for i in axp:
        fix[i] = True
    # modify literals according to fix array
    inst_ = inst.copy()
    for i in range(dDNNFman.nv):
        if not fix[i]:
            inst_[i] = None
    # 1) axp is a weak AXp
    if (pred and not dDNNFman.check_ICO_VA(inst_, va=True)) or \
            (not pred and not dDNNFman.check_ICO_VA(inst_, va=False)):
        print(f'given axp {axp} is not a weak AXp')
        return False
    # 2) axp is subset-minimal
    for i in range(dDNNFman.nv):
        if fix[i]:
            fix[i] = not fix[i]
            inst_[i] = None
            if (pred and not dDNNFman.check_ICO_VA(inst_, va=True)) or \
                    (not pred and not dDNNFman.check_ICO_VA(inst_, va=False)):
                inst_[i] = inst[i]
                fix[i] = not fix[i]
            else:
                print(f'given axp {axp} is not subset-minimal')
                return False
    return True