import numpy as np
import pytest
import scipy as sp
import pandas as pd
from src.mlp3_data_generator import data_gen_mlp3


def test_data_gen_mlp3_initialization():
    mlp3_data = data_gen_mlp3(n=1, m=1, seed=42)
    assert mlp3_data.n == 1
    assert mlp3_data.m == 1
    mlp3_data.W = np.array([[1]])
    mlp3_data.v = np.array([[1]])
    result = mlp3_data.model['function'](np.array([[0]]))
    expected = np.array([[0.5]])
    assert result == pytest.approx(expected, rel=1e-5)
