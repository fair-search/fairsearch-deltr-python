'''
Created on Oct 3, 2017

@author: meike.zehlike

'''
import numpy as np
import pandas as pd
import random, uuid
import itertools


class SyntheticDatasetCreator(object):

    """
    a dataframe that contains protected and non-protected features in columns. Each row represents
    a candidate with their feature values
    """
    @property
    def dataset(self):
        return self.__dataset


    """
    refers to possible combinations of protected attributes. Each group is an element of the Cartesian
    product of the element set per protected attribute.
    example:   attribute gender has two possible elements {0, 1}, attribute ethnicity has three
               possible elements {0, 1, 2} --> there are six groups
               a group is determined by one of the tuples (0, 0), (0,1), (1, 0), ..., (2, 1)
    the non-protected group is always represented by the tuple with only zeros
    """
    @property
    def groups(self):
        return self.__groups


    def __init__(self, size, attributeNamesAndCategories, nonProtectedAttributes):
        """
        @param size {int} :                         How many data points shall be created
        @param attributeNamesAndCategories {dict} : key = name of protected attribute; value = int that 
                                                    tells how many manifestations an attribute can have
        @param nonProtectedAttributes:              a string array that contains the names of the 
                                                    non-protected features
        """
        self.__dataset = pd.DataFrame()

        # determine groups of candidates
        self.__determineGroups(attributeNamesAndCategories)

        # generate distribution of protected attributes
        self.__createCategoricalProtectedAttributes(attributeNamesAndCategories, size)

        # generate scores per group
        self.__createScoresNormalDistribution(nonProtectedAttributes)
#        self.__createScoresUniformDistribution(nonProtectedAttributes)


        # generate ID column
        # self.__dataset['uuid'] = uuid.uuid4()


    def writeToJSON(self, path):
        self.__dataset.to_json(path, orient='records', lines=True)


    def writeToTXT(self, path):
        self.__dataset.to_csv(path, index=False, header=False)


    def __determineGroups(self, attributeNamesAndCategories):
        elementSets = []
        for attr, cardinality in attributeNamesAndCategories.items():
            elementSets.append(list(range(0, cardinality)))

        self.__groups = list(itertools.product(*elementSets))


    def __createScoresNormalDistribution(self, nonProtectedAttributes):
        """
        @param nonProtectedAttributes:     a string array that contains the names of the non-protected
                                           features
        """
        # if len(mu_diff) != len(nonProtectedAttributes) or len(sigma_diff) != len(nonProtectedAttributes):
        #    raise ValueError("lengths of arrays nonProtectedAttributes, mu_diff and sigma_diff have to match")

        def score(x, colName):
            mu = np.random.uniform()
            sigma = np.random.uniform()
            x[colName] = np.random.normal(mu, sigma, size=len(x))
            return x

        for attr in nonProtectedAttributes:
            self.__dataset = self.__dataset.groupby(self.__dataset.columns.tolist(), as_index=False,
                                                    sort=False).apply(score, (attr))


    def __createScoresUniformDistribution(self, nonProtectedAttributes):
        """
        @param nonProtectedAttributes:     a string array that contains the names of the non-protected
                                           features
        """

        def score(x, colName):
            highest = np.random.uniform()
            x[colName] = np.random.uniform(high=highest, size=x.size)
            return x

        for attr in nonProtectedAttributes:
            self.__dataset = self.__dataset.groupby(self.__dataset.columns.tolist(), as_index=False,
                                                    sort=False).apply(score, (attr))


    def __createScoresNormalDistributionGroupsSeparated(self, size):
            """
            @param size: expected size of the dataset
            """

            prot_data = pd.DataFrame()
            prot_data['gender'] = np.ones(int(size / 2)).astype(int)
            prot_data['score'] = np.random.normal(0.2, 0.3, size=int(size / 2))


            nonprot_data = pd.DataFrame()
            nonprot_data['gender'] = np.zeros(int(size / 2)).astype(int)
            nonprot_data['score'] = np.random.normal(0.8, 0.3, size=int(size / 2))

            self.__dataset = pd.concat([prot_data, nonprot_data])

            # normalize data
            mini = self.__dataset['score'].min()
            maxi = self.__dataset['score'].max()
            self.__dataset['score'] = (self.__dataset['score'] - mini) / (maxi - mini)



    def __createScoresUniformDistributionGroupsSeparated(self, size):
            """
            @param size:     expected size of the dataset
            """

            prot_data = pd.DataFrame()
            prot_data['gender'] = np.ones(int(size / 2)).astype(int)
            prot_data['score'] = np.random.uniform(high=0.5, low=0.0, size=int(size / 2))

            nonprot_data = pd.DataFrame()
            nonprot_data['gender'] = np.zeros(int(size / 2)).astype(int)
            nonprot_data['score'] = np.random.uniform(high=1.0, low=0.5, size=int(size / 2))

            self.__dataset = pd.concat([prot_data, nonprot_data])

    def __createCategoricalProtectedAttributes(self, attributeNamesAndCategories, numItems):
        """
        @param attributeNamesAndCategories:         a dictionary that contains the names of the
                                                    protected attributes as keys and the number of
                                                    categories as values
                                                    (e.g. {('ethnicity'; 5), ('gender'; 2)})
        @param numItems:                            number of items in entire created dataset (all
                                                    protection status)

        @return category zero is assumed to be the non-protected
        """
        newData = pd.DataFrame(columns=attributeNamesAndCategories.keys())

        for attributeName in attributeNamesAndCategories.keys():
            col = []
            categories = range(0, attributeNamesAndCategories[attributeName])
            for count in range(0, numItems):
                col.append(random.choice(categories))
            newData[attributeName] = col

        # add protected columns to dataset
        self.__dataset = self.__dataset.append(newData)


if __name__ == '__main__':
    # sdc = SyntheticDatasetCreator(20, {'gender': 2}, ['age', 'education', 'ethnicity'])
    #     #
    #     # print(sdc.dataset)
    from fairsearchdeltr import deltr

    data = pd.read_csv(r'D:\Projects\Meike-FairnessInL2R-Code\data\testdaten.csv', decimal=',')

    data['doc_id'] = pd.Series(range(50))

    data = data[['id', 'doc_id', 'gender', 'score', 'judgment']]

    d = deltr.Deltr(0 - 2, 1, 100)

    print(d.train(data))

    series = pd.Series([int(random.random() / 0.5) for r in range(50)])

    data2 = data.copy()
    data2['gender'] = series

    omega = [-0.1861255,   0.29714501]

    data3 = d.rank(data2)

    print(np.asarray(data.loc[:, 'gender']))
    print(np.asarray(data2.loc[:,'gender']))
    print(np.asarray(data3.loc[:, 'gender']))

    print(d._log)

    """
        gender       age  education  ethnicity
0        0  0.333053   0.875881   0.211208
1        0  0.216918  -0.861458   0.854239
2        0 -0.047481   0.461664   0.857284
3        1  1.694802   1.514378   0.517619
4        0  0.112723  -0.139726   0.583634
5        0  0.454562   1.912711   0.320170
6        1 -0.064455   0.705569   0.765517
7        1  0.083254   0.749731   0.228122
8        0 -0.087719   1.093907   0.656253
9        0  0.119039   0.221224  -0.054818
10       1 -0.415478   0.906090   0.283278
11       0  0.231553   0.566803   0.741297
12       0  0.070365   0.751321   1.650792
13       0  0.229923   0.537356   0.961001
14       0  0.222157   0.161757   0.654201
15       1  1.430287  -0.148758   1.826677
16       0  0.448757   0.904154   1.012863
17       1 -0.374923   0.310922   0.221493
18       0  0.183393   0.175672   0.321441
19       1  0.905846   0.901847   0.957311
    """