# -*- coding: utf-8 -*-

"""
fairsearchdeltr.models
~~~~~~~~~~~~~~~
This module contains the primary objects that power fairsearchdeltr.
"""


class TrainStep(object):
    """The :class:`TrainStep` object, which is a representation of the parameters in each step of the training.
    Contains a `timestamp`, `omega`, `omega_gradient`, `loss`, `loss_standard`, `loss_exposure`.
    TODO: is the name of the class OK?
    """
    def __init__(self, timestamp, omega, omega_gradient, loss_standard, loss_exposure, loss):
        """
        Object constructor
        :param timestamp:           timestamp of object creation
        :param omega:               current values of model
        :param omega_gradient:      calculated gradient
        :param loss_standard:       represents the change in ranking of training set vs predictions for training set
        :param loss_exposure:       represents the difference in exposures
        :param loss:                this should decrease at each iteration of training
        """
        self.timestamp = timestamp
        self.omega = omega
        self.omega_gradient = omega_gradient
        self.loss_standard = loss_standard
        self.loss_exposure = loss_exposure
        self.loss = loss

    def __repr__(self):
        return "<TrainStep [{0},{1},{2},{3},{4}]>".format(self.timestamp, self.omega, self.omega_gradient,
                                     self.loss_standard, self.loss_exposure)


class FairScoreDoc(object):
    """The :class:`FairScoreDoc` object, which is a representation of the items in the rankings.
    Contains an `id`, `is_protected` attribute, `features` and a `score`
    """

    def __init__(self, id, is_protected, features, score):
        self.id = id
        self.score = score
        self.features = features
        self.is_protected = is_protected

    def __repr__(self):
        return "<FairScoreDoc [%s]>" % ("Protected" if self.is_protected else "Nonprotected")


class Query(object):
    """The :class:`FairScoreDoc` object, which is a representation of the items in the rankings.
        Contains an `id`, `is_protected` attribute, `features` and a `score`
    """

    def __init__(self, id, docs):
        self.id = id
        self.docs = docs