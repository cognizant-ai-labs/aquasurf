
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
"""
Directed Acyclic Graph Activation Function
"""
import math
import numpy as np
from scipy import integrate
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras.layers import Layer # pylint: disable=import-error, no-name-in-module

N_ARY_FUNCTIONS = {
    'sum_n' : lambda inputs : tf.math.reduce_sum(inputs, axis=0),
    'prod_n' : lambda inputs : tf.math.reduce_prod(inputs, axis=0),
    'max_n' : lambda inputs : tf.math.reduce_max(inputs, axis=0),
    'min_n' : lambda inputs : tf.math.reduce_min(inputs, axis=0),
}

BINARY_FUNCTIONS = {
    'add' : tf.math.add,
    'sub' : tf.math.subtract,
    'mul' : tf.math.multiply,
    'div' : tf.math.divide,
    'pow' : tf.math.pow,
    'max' : tf.math.maximum,
    'min' : tf.math.minimum,
}

UNARY_FUNCTIONS = {
    'zero' : tf.zeros_like,
    'one' : tf.ones_like,
    'identity' : tf.identity,
    'negative' : tf.math.negative,

    'abs' : tf.abs,
    'reciprocal' : tf.math.reciprocal,
    'square' : tf.square,
    'exp' : tf.math.exp,

    'erf' : tf.math.erf,
    'erfc' : tf.math.erfc,
    'sinh' : tf.sinh,
    'cosh' : tf.cosh,

    'tanh' : tf.tanh,
    'expm1' : tf.math.expm1,
    'sigmoid' : tf.sigmoid,
    'log_sigmoid' : tf.math.log_sigmoid,

    'arcsinh' : tf.math.asinh,
    'arctan' : tf.math.atan,
    'bessel_i0e' : tf.math.bessel_i0e,
    'bessel_i1e' : tf.math.bessel_i1e,

    'relu' : tf.nn.relu,
    'elu' : tf.nn.elu,
    'selu' : tf.nn.selu,
    'swish' : tf.nn.swish,

    'softplus' : tf.nn.softplus,
    'softsign' : tf.nn.softsign,
    'hard_sigmoid' : tf.keras.activations.hard_sigmoid,

    'elish' : lambda x : tf.where(tf.greater(x,0), x / (1 + tf.math.exp(-x)), (tf.math.exp(x)-1) / (1 + tf.math.exp(-x))),
    'leakyrelu' : lambda x : tf.where(tf.greater(x, 0), x, 0.01 * x),
    'mish' : lambda x : x * tf.math.tanh(tf.keras.activations.softplus(x)),
    'gelu' : tf.keras.activations.gelu,
}

FUNCTIONS = {**N_ARY_FUNCTIONS, **BINARY_FUNCTIONS, **UNARY_FUNCTIONS}

INF = float('inf')

def gaussian_mean(afn):
    return integrate.quad(lambda x : afn(x) * norm.pdf(x), -INF, INF)[0]

def second_moment(afn):
    return integrate.quad(lambda x : math.pow(afn(x), 2) * norm.pdf(x), -INF, INF)[0]

class ActivationFunction(Layer):
    """
    This class implements custom activation functions.  They are constructed by providing
    a string that specifies the function.  The string is parsed and the function is
    constructed as a directed acyclic graph of tensorflow operations.  The resulting
    ActivationFunction can be used as an ordinary TensorFlow Layer.  For example:

    x = ActivationFunction(fn_name='max(relu(x),cosh(elu(x)))')(x)

    or

    x = ActivationFunction(fn_name='sum_n(abs(x),swish(x),sigmoid(x))')(x)

    If normalize=True is passed, the function is modified so that the output has
    mean zero and variance one, in expectation.
    """
    def __init__(self, fn_name, normalize=False, **kwargs):
        super().__init__(**kwargs)
        # remove whitespace
        self.fn_name = fn_name.replace(' ', '')
        self.callable_fn = None
        self.normalize = normalize

    def construct_function(self, fn_name):
        """
        Parse the function name and construct the function as a directed acyclic graph
        """
        if fn_name == 'x':
            # we are at the input
            return tf.identity

        # operation name is before the first '('
        # args are within the first '(' and the last ')'
        first_paren = fn_name.index('(')
        last_paren = fn_name.rindex(')')
        op_name = fn_name[:first_paren]
        args = fn_name[first_paren+1:last_paren]

        # args are separated by commas not nested within parentheses
        indices_to_split = []
        for i, char in enumerate(args):
            if char == ',' and \
                args[:i].count('(') == args[:i].count(')') and \
                    args[i:].count('(') == args[i:].count(')'):
                indices_to_split.append(i)
        indices_to_split.append(len(args))
        arg_names = [args[i+1:j] for i, j in zip([-1] + indices_to_split, indices_to_split)]

        operation = FUNCTIONS[op_name]
        args = [self.construct_function(arg_name) for arg_name in arg_names]

        if op_name in N_ARY_FUNCTIONS:
            return lambda x : operation(tf.stack([arg(x) for arg in args]))

        if op_name in BINARY_FUNCTIONS:
            assert len(args) == 2
            return lambda x : operation(args[0](x), args[1](x))

        assert op_name in UNARY_FUNCTIONS
        assert len(args) == 1
        arg = args[0]
        return lambda x : operation(arg(x))

    def build(self, input_shape): # pylint: disable=unused-argument
        """
        Build the function
        """
        callable_fn = self.construct_function(self.fn_name)
        if self.normalize:
            gauss_mean = gaussian_mean(callable_fn)
            centered_fn = lambda x : callable_fn(x) - gauss_mean
            sec_moment = second_moment(centered_fn)
            normalized_fn = lambda x : centered_fn(x) / np.sqrt(sec_moment)
            self.callable_fn = normalized_fn
        else:
            self.callable_fn = callable_fn

    def call(self, inputs):
        """
        Call the function
        """
        return self.callable_fn(inputs)

    def get_config(self):
        """
        Get the config.  Used for serialization.
        """
        config = super().get_config()
        config.update({'fn_name'   : self.fn_name,
                       'normalize' : self.normalize})
        return config
