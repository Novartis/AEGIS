#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""errors.py

The file containing all of the project's specific error exceptions

"""


class SamplingSizeError(Exception):
    """Raised when sampling without replacement cannot be done"""

    pass


class FormatError(Exception):
    """Raised when the format of a dataframe is unexpected"""

    pass
