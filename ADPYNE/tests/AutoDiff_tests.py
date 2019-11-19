import warnings
import pytest
import numpy as np
from AutoDiff import AutoDiff

# helper function tests

# _convertNonArray

# _calcJacobian

# addition tests
def test_add_ad_results():
	# single input cases
	# positive numbers
	x = AutoDiff(5, 2)
	f = x + x
	assert f.val == 10
	assert f.der == 4
	assert f.jacobian == 2
	# negative numbers
	y = AutoDiff(-5, 2)
	f = y + y
	assert f.val == -10
	assert f.der == 4
	assert f.jacobian == 2

def test_add_constant_results():
	# single input case
	# positive numbers
	x = AutoDiff(5, 2)
	f = x + 3
	assert f.val == 8
	assert f.der == 2
	assert f.jacobian == 1
	# negative numbers
	x = AutoDiff(-5, 2)
	f = x + 3
	assert f.val == -2
	assert f.der == 2
	assert f.jacobian == 1

# reverse addition tests
def test_add_constant_results():
	# single input case
	# positive numbers
	x = AutoDiff(5, 2)
	f = 3 + x
	assert f.val == 8
	assert f.der == 2
	assert f.jacobian == 1
	# negative numbers
	x = AutoDiff(-5, 2)
	f = 3 + x
	assert f.val == -2
	assert f.der == 2
	assert f.jacobian == 1
# subtraction tests

# reverse subtraction tests

# multiplication tests

# reverse multiplication tests

# division tests

# reverse division tests

# power tests

# reverse power tests

# positive tests

# negation tests

# absolute value tests

# invert tests