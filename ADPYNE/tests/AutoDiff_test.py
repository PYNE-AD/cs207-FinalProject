import warnings
import pytest
import numpy as np
# from ADPYNE import AutoDiff
# from ADPYNE import elemFunctions as ef
from ADPYNE import AutoDiff
from ADPYNE import elemFunctions as ef

# helper function tests

def test_convertNonArray_array():
    AD = AutoDiff(np.array([[1,2]]),1)
    assert np.all(np.equal(AD.val, np.array([[1,2]])))
    

def test_convertNonArray_num():
    AD = AutoDiff(1,1)
    assert np.all(np.equal(AD.val, np.array([[1]])))


def test_calcJacobian_array():
    AD = AutoDiff(1,2)
    assert np.all(np.equal(AD.jacobian, np.array([[1]])))
    
    
def test_calcJacobian_array_withJ():
    AD = AutoDiff(1,1,1,0,np.array([[1]]))
    assert np.all(np.equal(AD.jacobian, np.array([[1]])))

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
def test_radd_constant_results():
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
def test_sub_ad_results():
    # single input cases
	# positive numbers
	x = AutoDiff(5, 2)
	f = x - x
	assert f.val == 0
	assert f.der == 0
	assert f.jacobian == 0
	# negative numbers
	y = AutoDiff(-5, 2)
	f = y - y
	assert f.val == 0
	assert f.der == 0
	assert f.jacobian == 0

def test_sub_constant_results():
    # single input case
	# positive numbers
	x = AutoDiff(5, 2)
	f = x - 3
	assert f.val == 2
	assert f.der == 2
	assert f.jacobian == 1
	# negative numbers
	x = AutoDiff(-5, 2)
	f = x - 3
	assert f.val == -8
	assert f.der == 2
	assert f.jacobian == 1

# reverse subtraction tests
def test_rsub_constant_results():
    # single input case
	# positive numbers
	x = AutoDiff(5, 2)
	f = 3 - x
	assert f.val == -2
	assert f.der == 2
	assert f.jacobian == 1
	# negative numbers
	x = AutoDiff(-5, 2)
	f = 3 - x
	assert f.val == 8
	assert f.der == 2
	assert f.jacobian == 1

# multiplication tests
def test_mul_ad_results():
    # single input case
    # positive numbers
    x = AutoDiff(5, 2)
    f = x * x
    assert f.val == 25
    assert f.der == 20
    assert f.jacobian == 10
    # negative numbers
    x = AutoDiff(-5, 2)
    f = x * x
    assert f.val == 25
    assert f.der == -20
    assert f.jacobian == -10

def test_mul_constant_results():
    # single input case
    # positive numbers
    x = AutoDiff(5, 2)
    f = x * 3
    assert f.val == 15
    assert f.der == 6
    assert f.jacobian == 3
    # negative numbers
    x = AutoDiff(-5, 2)
    f = x * 3
    assert f.val == -15
    assert f.der == 6
    assert f.jacobian == 3

# reverse multiplication tests
def test_rmul_constant_results():
    # single input case
    # positive numbers
    x = AutoDiff(5, 2)
    f = 3 * x
    assert f.val == 15
    assert f.der == 6
    assert f.jacobian == 3
    # negative numbers
    x = AutoDiff(-5, 2)
    f = 3 * x
    assert f.val == -15
    assert f.der == 6
    assert f.jacobian == 3

# division tests
def test_truediv_ad_results():
    # single input case
    # positive numbers
    x = AutoDiff(5, 2)
    f = x / x
    assert f.val == 1
    assert f.der == 0
    assert f.jacobian == 0
    # negative numbers
    x = AutoDiff(-5, 2)
    f = x / x
    assert f.val == 1
    assert f.der == 0
    assert f.jacobian == 0

def test_truediv_constant_results():
    # single input case
    # positive numbers
    x = AutoDiff(9, 6)
    f = x / 3
    assert f.val == 3
    assert f.der == 2
    assert f.jacobian == (1/3)
    # negative numbers
    x = AutoDiff(-9, 6)
    f = x / 3
    assert f.val == -3
    assert f.der == 2
    assert f.jacobian == (1/3)

# reverse division tests
def test_rtruediv_constant_results():
    # single input case
    # positive numbers
    x = AutoDiff(3, 2)
    f = 6 / x
    assert f.val == 2
    assert f.der == 3
    assert f.jacobian == 6
    # negative numbers
    x = AutoDiff(-3, 2)
    f = 6 / x
    assert f.val == -2
    assert f.der == 3
    assert f.jacobian == 6

# power tests
def test_pow_ad_results():
    x = AutoDiff(2, 1)
    f = x**x
    assert f.val == 4
    assert f.der == 4 + np.log(16)
    assert f.jacobian == 4 + np.log(16)

def test_pow_constant_results():
    # positive numbers
    x = AutoDiff(5, 2)
    f = x**3
    assert f.val == 125
    assert f.der == 150
    assert f.jacobian == 75
    # negative numbers
    x = AutoDiff(-5, 2)
    f = x**3
    assert f.val == -125
    assert f.der == 150
    assert f.jacobian == 75

# reverse power tests
def test_rpow_constant_results():
    x = AutoDiff(5, 2)
    f = 3**x
    assert f.val == 243
    assert f.der == 486 * np.log(3)
    assert f.jacobian == 243 * np.log(3)

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