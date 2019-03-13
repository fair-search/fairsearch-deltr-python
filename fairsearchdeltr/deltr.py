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
        if protected_feature is None:
            raise ValueError("The index of column in data `protected_feature` must be initialized")
        if gamma is None:
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
        self._loss = None

    def train(self, training_set: pd.DataFrame):
        """
        Trains a DELTR model on a given training set
        :param training_set:        requires first column to contain the query ids, second column the document ids
                                    and last column to contain the training judgments in descending order
                                    i.e. higher scores are better
        :return:                    returns the model
        """

        # create the trainer
        tr = trainer.Trainer(self._protected_feature, self._gamma, self._number_of_iterations, self._learning_rate,
                             self._lambda, self._init_var)

        # prepare data
        query_ids, doc_ids, protected_attributes, feature_matrix, training_scores = prepare_data(training_set,
                                                                                                 self._protected_feature)

        # launch training routine
        self._omega, self._loss = tr.train_nn(query_ids, feature_matrix, training_scores)

        # return model
        return self._omega

    def rank(self, prediction_set: pd.DataFrame):
        """
        Uses the trained DELTR model to rank the prediction set
        :param prediction_set:      requires first column to contain the query ids, second column the document ids
                                    and (optionally) last column to contain the training judgments in descending order
                                    i.e. higher scores are better
        :return:                    returns the model
        """

        if self._omega is None:
            raise SystemError("You need to train a model first!")

        # prepare data
        query_ids, doc_ids, protected_attributes,feature_matrix, initial_scores = prepare_data(prediction_set,
                                                                                               self._protected_feature)

        # calculate the predictions
        predictions = np.dot(feature_matrix, self._omega)

        # create the resulting data frame
        result = pd.DataFrame({'doc_id': doc_ids, 'gender': protected_attributes, 'score': predictions})

        # sort by the score in descending order
        result = result.sort_values(['score'], ascending=[0])

        return result

    def loss(self):
        if self._omega is None:
            raise SystemError("You need to train a model first!")
        return self._loss


def prepare_data(data, protected_column):
    """
    Extracts the different columns of the input data
    :param data:
    :return:
    """
    query_ids = np.asarray(data.iloc[:, 0])
    doc_ids = np.asarray(data.iloc[:, 1])
    protected_attributes = np.asarray(data.iloc[:, protected_column + 2]) # add 2 for query id and doc id
    feature_matrix = np.asarray(data.iloc[:, 2:(data.shape[1] - 1)])
    scores = np.reshape(np.asarray(data.iloc[:, data.shape[1] - 1]),
                                 (feature_matrix.shape[0], 1))

    return query_ids, doc_ids, protected_attributes, feature_matrix, scores
