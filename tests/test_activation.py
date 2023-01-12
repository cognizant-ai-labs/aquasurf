
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

from unittest import TestCase
import numpy as np
from aquasurf.activation import ActivationFunction

class ActivationTest(TestCase):
    """
    Test that an activation function can be created and used
    """

    def setUp(self):
        self.fn_name = 'add(relu(x),swish(x))'
        self.inputs = np.random.random((100,))

    def test_activation(self):
        fn = ActivationFunction(self.fn_name)
        output = fn(self.inputs)
