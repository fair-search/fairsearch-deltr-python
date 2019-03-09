# -*- coding: utf-8 -*-

"""
fairsearchdeltr.deltr
~~~~~~~~~~~~~~~
This module serves as a wrapper around the utilities we have created for DELTR
"""

class Deltr():
    def __init__(self):
        raise NotImplementedError()

    def train(self, training_set, protected_feature, gamma, **kwargs):
        raise NotImplementedError()

    def rank(self, training_set, protected_feature, gamme, **kwargs):
        raise NotImplementedError()

    def loss(self, training_set, protected_feature, gamma, **kwargs):
        raise NotImplementedError()
