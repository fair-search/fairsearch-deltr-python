# -*- coding: utf-8 -*-

"""
fairsearchdeltr.ranker
~~~~~~~~~~~~~~~
This module holds the detailed mechanics of DELTR ranker (predictor)
"""

import numpy as np
import pandas as pd


class Predictor():
    '''
    :field __pathToTestData: string with path to test data file
    :field __pathToModelFile: string with path to model file, which contains feature weights
    :field __resultDir: string with directory into which results are stored
    :field __protectedColumn: int with index of column that contains protected attribute in test data
    :field __quiet: if True, no command line outputs are printed during calculation
    '''

    def __init__(self, model, protected_feature):
        self.model = model
        self.protected_feature = protected_feature

    def predict(self, features):
        omega = self.model
        predictions = np.dot(features, omega)
        return predictions

