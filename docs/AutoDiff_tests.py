import warnings
import pytest
import numpy as np
from AutoDiff import AutoDiff

# positive tests
def test_pos_results():
    # positive numbers
    x = AutoDiff(5, 2)
    f = + x
    assert f.val == 5
    assert f.der == 2
    assert f.jacobian == 1
    # negative numbers
    y = AutoDiff(-5, 2)
    f = + y
    assert f.val == -5
    assert f.der == 2
    assert f.jacobian == 1

# negation tests
def test_neg_results():
    # positive numbers
    x = AutoDiff(5, 2)
    f = - x
    assert f.val == -5
    assert f.der == -2
    assert f.jacobian == -1
    # negative numbers
    y = AutoDiff(-5, 2)
    f = - y
    assert f.val == 5
    assert f.der == -2
    assert f.jacobian == -1

def test_ned_constant_results():
    x = 3
    f = - x
    assert f == -3

# absolute value tests
def test_abs_results():
    # positive numbers
    x = AutoDiff(5, 2)
    f = abs(x)
    assert f.val == 5
    assert f.der == 2
    assert f.jacobian == 1
    # negative numbers
    y = AutoDiff(-5, 2)
    f = abs(y)
    assert f.val == 5
    assert f.der == -2
    assert f.jacobian == -1

def test_abs_constant_results():
    x = -3
    f = abs(x)
    assert f == 3


