
# Copyright (C) 2023 Cognizant Digital Business, Evolutionary AI.
# All Rights Reserved.
# Issued under the Academic Public License.
#
# You can be released from the terms, and requirements of the Academic Public
# License by purchasing a commercial license.
# Purchase of a commercial license is mandatory for any use of the
# AQuaSurF Software in commercial settings.
#
# END COPYRIGHT
from setuptools import setup, find_packages

setup(name='aquasurf',
    version='0.0.1',
    description='Efficient Activation Function Optimization through Surrogate Modeling',
    author='@garrettbingham',
    author_email='garrett.bingham@cognizant.com',
    packages=find_packages(include=['aquasurf', 'aquasurf.*']),
)
