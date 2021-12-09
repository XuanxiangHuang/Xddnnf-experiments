#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   SDD Classifiers explainer
#   Author: Xuanxiang Huang
#
################################################################################
import resource
from pysdd.sdd import SddNode
from pysat.formula import IDPool
from pysat.solvers import Solver
################################################################################


class XpSdd(object):
    """
        Explain SDD classifier.
    """
    def __init__(self, root: SddNode, nv, features, sup_vars, verb=0):
        self.root = root
        self.nv = nv
        self.features = features
        self.sup_vars = sup_vars
        self.verbose = verb

    def inst2lits(self, inst):
        """
            Pre-processing: THIS IS MANDATORY !!

            Given an instance (array of 0 and 1),
            return corresponding list of literals
            e.g. for variable x, x=0 means -x, x=1 means x
            Note that if variable x_i is don't cared, then
            then i-th position is None

            :param inst: given instance
            :return: list of literals
        """
        assert (self.nv == len(inst))
        lits = [None] * self.nv

        for j in range(self.nv):
            if self.sup_vars[j]:
                if int(inst[j]):
                    lits[j] = self.sup_vars[j].literal
                else:
                    lits[j] = -self.sup_vars[j].literal
        return lits

    def get_predict(self, lits):
        """
            Given a list of literals, return its prediction
            :param lits: given an instance
            :return:
        """
        # get prediction of instance
        out = self.root
        for item in lits:
            if item:
                out = out.condition(item)
        assert out.is_true() or out.is_false()
        return True if out.is_true() else False

    def reachable(self, lits, univ, pred):
        """
            Check if desired prediction/class is reachable.

            :param lits: given list of literals
            :param univ: list of universal features
            :param pred: desired prediction
            :return: True if reachable else False
        """
        lits_ = lits.copy()
        for i in range(self.nv):
            if univ[i]:
                lits_[i] = None

        tmp = self.root
        for item in lits_:
            if item:
                tmp = tmp.condition(item)
        if pred:
            return not tmp.is_false()
        else:
            return not tmp.is_true()

    def find_axp(self, lits, fixed=None):
        """
            Compute one abductive explanation (Axp).

            :param fixed: a list of features declared as fixed.
            :param lits: given list of literals.
            :return: one abductive explanation,
                        each element in the return Axp is a feature index.
        """
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime

        # get prediction of literals
        pred = self.get_predict(lits)
        # get/create fix array
        if not fixed:
            fix = [True for _ in range(self.nv)]
        else:
            fix = fixed.copy()
        assert (len(fix) == self.nv)
        # modify literals according to fix array
        lits_ = lits.copy()
        for i in range(self.nv):
            if not fix[i]:
                lits_[i] = None

        for i in range(self.nv):
            if fix[i]:
                fix[i] = not fix[i]
                lits_[i] = None
                tmp = self.root
                for item in lits_:
                    if item:
                        tmp = tmp.condition(item)
                if (pred and not tmp.is_true()) or (not pred and not tmp.is_false()):
                    lits_[i] = lits[i]
                    fix[i] = not fix[i]

        # axp is a subset of fixed features, and it is minimal
        axp = [i for i in range(self.nv) if fix[i]]

        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime - time

        if self.verbose:
            if self.verbose == 1:
                print(f"Axp: {axp}")
            elif self.verbose == 2:
                print(f"Axp: {axp} ({[self.features[i] for i in axp]})")
            print("Runtime: {0:.4f}".format(time))

        return axp

    def find_cxp(self, lits, universal=None):
        """
            Compute one contrastive explanation (Cxp).

            :param universal: a list of features declared as universal.
            :param lits: given list of literals.
            :return: one contrastive explanation,
                        each element in the return Cxp is a feature index.
        """
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime

        # get prediction of literals
        pred = self.get_predict(lits)
        # get/create univ array
        if not universal:
            univ = [True for _ in range(self.nv)]
        else:
            univ = universal.copy()
        assert (len(univ) == self.nv)
        # modify literals according to univ array
        lits_ = lits.copy()
        for i in range(self.nv):
            if univ[i]:
                lits_[i] = None

        for i in range(self.nv):
            if univ[i]:
                univ[i] = not univ[i]
                lits_[i] = lits[i]
                tmp = self.root
                for item in lits_:
                    if item:
                        tmp = tmp.condition(item)
                if (pred and tmp.is_true()) or (not pred and tmp.is_false()):
                    lits_[i] = None
                    univ[i] = not univ[i]

        # cxp is a subset of universal features, and it is minimal
        cxp = [i for i in range(self.nv) if univ[i]]

        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime - time

        if self.verbose:
            if self.verbose == 1:
                print(f"Cxp: {cxp}")
            elif self.verbose == 2:
                print(f"Cxp: {cxp} ({[self.features[i] for i in cxp]})")
            print("Runtime: {0:.4f}".format(time))

        return cxp

    def enum_exps(self, lits):
        """
            Enumerate all (abductive and contrastive) explanations, using MARCO algorithm.

            :param lits: given list of literals
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

        pred = self.get_predict(lits)

        num_axps = 0
        num_cxps = 0
        axps = []
        cxps = []

        slv = Solver(name="glucose3")
        for i in range(self.nv):
            new_var('u_{0}'.format(i))
        # initially all features are fixed
        univ = [False for _ in range(self.nv)]

        while slv.solve():
            # first model is empty
            model = slv.get_model()
            for lit in model:
                name = vpool.obj(abs(lit)).split(sep='_')
                univ[int(name[1])] = False if lit < 0 else True
            if self.reachable(lits, univ, not pred):
                cxp = self.find_cxp(lits, univ)
                slv.add_clause([-new_var('u_{0}'.format(i))
                                for i in cxp])
                num_cxps += 1
                cxps.append(cxp)
            else:
                fix = [not i for i in univ]
                axp = self.find_axp(lits, fix)
                slv.add_clause([new_var('u_{0}'.format(i))
                                for i in axp])
                num_axps += 1
                axps.append(axp)

        slv.delete()

        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime - time

        print('#AXp:', num_axps)
        print('#CXp:', num_cxps)
        print("Runtime: {0:.4f}".format(time))

        return axps, cxps
