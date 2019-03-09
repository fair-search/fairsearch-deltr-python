# -*- coding: utf-8 -*-

"""
---------------------------------------
Fair search DELTR for Python
---------------------------------------

A Python library for disparate exposure in ranking (a learning to rank approach)


For details on support methods - see `fairsearchdeltr.fair` and `fairsearchdeltr.simulator`. Full documentation
is at https://github.com/fair-search/fairsearchdeltr-python.

:copyright: (c) 2019 by Ivan Kitanovski
:license: Apache 2.0, see LICENSE for more details.
"""

# Set default logging handler to avoid "No handler found" warnings.
import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())