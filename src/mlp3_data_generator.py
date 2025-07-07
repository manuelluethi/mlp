# In this module, we define collections of data sets which we will use for
# testing speed of convergence and quality of approximation of MLP with a
# single hidden layer
import numpy as np
import pandas as pd
import scipy as sp
from types import MappingProxyType


class data_gen_mlp3():
    # Constructor
    # Input:    self = instance parameter
    #           seed = random seed for reproducibility
    def __init__(self, n, m, seed=None):
        # set random seed for reproducibility
        self._seed = 42
        # If seed was passed as argument, update
        if seed is not None:
            self._seed = seed
        # define a module specific random number generator
        self._rng = np.random.default_rng(self._seed)
        # Initialize MLP parameters as wealy private to force updates
        self._n = n
        self._m = m
        self._W = self._rng.uniform(-1 / n, 1 / n, (m, n))  # mxn
        self._v = self._rng.uniform(-1 / m, 1 / m, (1, m))  # 1xm
        self._make_model()  # construct model

    def _make_model(self):
        def func(feature):
            hidden = sp.special.expit(np.matmul(self._W, feature))  # mx1
            output = np.matmul(self._v, hidden)  # 1x1
            return output

        self.model = MappingProxyType({
            "function": func,
            "parameters": MappingProxyType({
                "n": self._n,
                "m": self._m,
                "W": self._W,
                "v": self._v
            }),
            "description": (
                "three layer perceptron with input layer of width "
                f"n = {self._n}, hidden layer of width m = {self._m}. "
                "The weight matrix for the hidden layer is W and for the "
                "output layer is v; cf. model.parameters.keys"
            )
        })

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        self._n = value
        self.W = self._rng.uniform(
            -1 / value,
            1 / value,
            (self.m, value)
        )
        self._make_model()  # Update the model when n changes

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, value):
        self._m = value
        self.v = self._rng.uniform(-1 / value, 1 / value, (1, value))
        # Update the weight matrix W when m changes
        self.W = self._rng.uniform(
            -1 / self._n,
            1 / self._n,
            (value, self._n)
        )
        self._make_model()  # Update the model when m changes

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, value):
        if value.shape != (self.m, self.n):
            raise ValueError(f"W must have shape ({self.m}, {self.n})")
        self._W = value
        self._make_model()  # Update the model when W changes

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        if value.shape != (1, self.m):
            raise ValueError(f"v must have shape (1, {self.m})")
        self._v = value
        self._make_model()  # Update the model when v changes


class mlp3_data():
    # Constructor
    # Input:    self = instance parameter
    #           seed = random seed for reproducibility
    #           model = the data generation model
    def __init__(self, seed=None, model=None):
        # set random seed for reproducibility
        self._seed = 42
        # If seed was passed as argument, update
        if seed is not None:
            self._seed = seed
        # define a module specific random number generator
        self.rng = np.random.default_rng(self._seed)
        self.df = pd.DataFrame()
        self.array = np.array([[]])
        if model is None:
            self.data_gen = data_gen_mlp3(1, 1, seed=self._seed)
            self.model = self.data_gen.model
        else:
            self.model = model

    # Generate data
    # Input:    self = instance parameters
    #           s = sample size
    def generate_data(self, s, n=None, m=None):
        # update model parameters if n or m are passed
        if n is not None:
            self.data_gen.n = n
        if m is not None:
            self.data_gen.m = m
        # update model
        if n is not None or m is not None:
            self.model = self.data_gen.model
        # Generate data using the model function
        # get width of input layer (which is n if passed as an argument)
        n = self.data_gen.n
        # each feature is a realization of n i.i.d. standard normal variables
        feat_sample = self.rng.standard_normal(size=(n, s))  # nxs
        # labels are determined by the model:we
        lab_sample = np.array(
            [
                self.model['function'](feat_sample[:, i])
                for i in range(s)
            ]
        ).reshape(1, s)     # 1xs
        self.array = np.vstack((feat_sample, lab_sample))
        self.df = pd.DataFrame(self.array.T)
        self.df.columns = [f"feature{i + 1}" for i in range(n)] + ["labels"]
