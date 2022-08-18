# -*- coding: utf-8 -*-
"""Setup script."""
from setuptools import find_packages, setup

setup(
    # mandatory
    name="mhciipresentation",
    # mandatory
    version="0.1",
    # mandatory
    author="Philip Hartout, Christian Schleberger, Celia Mendez-Garcia",
    packages=find_packages(),
    setup_requires=["isort"],
)
