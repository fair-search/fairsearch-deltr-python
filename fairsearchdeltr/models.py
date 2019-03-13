# -*- coding: utf-8 -*-

"""
fairsearchdeltr.models
~~~~~~~~~~~~~~~
This module contains the primary objects that power fairsearchdeltr.
"""


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