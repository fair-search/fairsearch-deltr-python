# -*- coding: utf-8 -*-

"""
fairsearchdeltr.deltr
~~~~~~~~~~~~~~~
This module serves as a wrapper around the utilities we have created for DELTR
"""

import numpy as np
import datetime
import pandas as pd

from fairsearchdeltr import trainer


class Deltr(object):

    def __init__(self, protected_feature: int, gamma: float, number_of_iterations=3000, learning_rate=0.001,
                 lambdaa=0.001, init_var=0.01):
        """
         Disparate Exposure in Learning To Rank
        --------------------------------------

        A supervised learning to rank algorithm that incorporates a measure of performance and a measure
        of disparate exposure into its loss function. Trains a linear model based on performance and
        fairness for a protected group.
        By reducing disparate exposure for the protected group, increases the overall group visibility in
        the resulting rankings and thus prevents systematic biases against a protected group in the model,
        even though such bias might be present in the training data.

        :param protected_feature:       index of column in data that contains protected attribute
        :param gamma:                   gamma parameter for the cost calculation in the training phase
                                        (recommended to be around 1)
        :param number_of_iterations     number of iteration in gradient descent (optional)
        :param learning_rate            learning rate in gradient descent (optional)
        :param lambdaa                  regularization constant (optional)
        :param init_var                 range of values for initialization of weights (optional)
        """

        # check if mandatory parameters are present
        if not protected_feature:
            raise ValueError("The index of column in data `protected_feature` must be initialized")
        if not gamma:
            raise ValueError("The `gamma` parameter must be initialized")

        # assign mandatory parameters
        self._protected_feature = protected_feature
        self._gamma = gamma

        # assign optional parameters
        self._number_of_iterations = number_of_iterations
        self._learning_rate = learning_rate
        self._lambda = lambdaa
        self._init_var = init_var

        # default init
        self._omega = None

    def train(self, training_set: pd.DataFrame):
        """
        Trains a DELTR model on a given training set
        :param training_set:        requires first column to contain the query ids and last column to contain the
                                    training judgments (in descending order, i.e. higher scores are better
        :return:                    returns the model
        """

        # create the trainer
        tr = trainer.Trainer(self._protected_feature, self._gamma, self._number_of_iterations, self._learning_rate,
                             self._lambda, self._init_var)

        # prepare data
        query_ids = np.asarray(training_set.iloc[:, 0])
        feature_matrix = np.asarray(training_set.iloc[:, 1:(training_set.shape[1] - 1)])
        training_scores = np.reshape(np.asarray(training_set.iloc[:, training_set.shape[1] - 1]),
                                     (feature_matrix.shape[0], 1))

        # launch training routine
        self._omega = tr.train_nn(query_ids, feature_matrix, training_scores)

        # return model
        return self._omega

    def rank(self, prediction_set):
        raise NotImplementedError()

    def loss(self, training_set, protected_feature, gamma, ):
        raise NotImplementedError()
