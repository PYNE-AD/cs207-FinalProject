import warnings
import pytest
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from ADPYNE.AutoDiff import AutoDiff, vectorize
import ADPYNE.elemFunctions as ef

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

def test_calcJacobian_vector():
	AD = AutoDiff(4, np.array([[2, 1]]).T, n=2, k=1)
	assert np.all(np.equal(AD.jacobian, np.array([[1, 0]])))
	AD = AutoDiff(3, np.array([[1, 2]]).T, n=2, k=2)
	assert np.all(np.equal(AD.jacobian, np.array([[0, 1]])))

def test_calcDerivative():
	AD = AutoDiff(4, 2, n=4, k=3)
	assert np.all(np.equal(AD.der, np.array([[0, 0, 2, 0]])))

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

def test_add_vector_results():
	x = AutoDiff(np.array([[3],[1]]), np.array([[2, 1]]).T, 2, 1)
	y = AutoDiff(np.array([[2],[-3]]), np.array([[3, 2]]).T, 2, 2)
	f = x + y
	assert np.all(f.val == np.array([[5], [-2]]))
	assert np.all(f.der == np.array([[2, 3], [1, 2]]))
	assert np.all(f.jacobian == np.array([[1, 1], [1, 1]]))

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

def test_add_constant_vector_results():
	x = AutoDiff(np.array([[1, 3]]).T, np.array([[2, 1]]).T, 2, 1)
	f = x + 3
	assert np.all(f.val == np.array([[4, 6]]).T)
	assert np.all(f.der == np.array([[2, 0], [1, 0]]))
	assert np.all(f.jacobian == np.array([[1, 0], [1, 0]]))

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

def test_radd_constant_vector_results():
	x = AutoDiff(np.array([[1, 3]]).T, np.array([[2, 1]]).T, 2, 1)
	f = 3 + x
	assert np.all(f.val == np.array([[4, 6]]).T)
	assert np.all(f.der == np.array([[2, 0], [1, 0]]))
	assert np.all(f.jacobian == np.array([[1, 0], [1, 0]]))

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

def test_sub_vector_results():
	x = AutoDiff([3, 1], [2, 1], 2, 1)
	y = AutoDiff([2, -3], [3, 2], 2, 2)
	f = x - y
	assert np.all(f.val == np.array([[1], [4]]))
	assert np.all(f.der == np.array([[2, -3], [1, -2]]))
	assert np.all(f.jacobian == np.array([[1, -1], [1, -1]]))

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

def test_sub_constant_vector_results():
	x = AutoDiff([1, 3], [2, 1], 2, 1)
	f = x - 3
	assert np.all(f.val == np.array([[-2, 0]]).T)
	assert np.all(f.der == np.array([[2, 0], [1, 0]]))
	assert np.all(f.jacobian == np.array([[1, 0], [1, 0]]))

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

def test_rsub_constant_vector_results():
	x = AutoDiff([1, 3], [2, 1], 2, 1)
	f = 3 - x
	assert np.all(f.val == np.array([[2, 0]]).T)
	assert np.all(f.der == np.array([[2, 0], [1, 0]]))
	assert np.all(f.jacobian == np.array([[1, 0], [1, 0]]))

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

def test_mul_vector_results():
	x = AutoDiff([3, 1], [2, 1], 2, 1)
	y = AutoDiff([2, -3], [1, 2], 2, 2)
	f = x*y
	assert np.all(f.val == np.array([[6, -3]]).T)
	assert np.all(f.der == np.array([[4, 3], [-3, 2]]))
	assert np.all(f.jacobian == np.array([[2, 3], [-3, 1]]))

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

def test_mul_constant_vector_results():
	x = AutoDiff([3, 1], [2, 1], 2, 1)
	f = x * 3
	assert np.all(f.val == np.array([[9, 3]]).T)
	assert np.all(f.der == np.array([[6, 0], [3, 0]]))
	assert np.all(f.jacobian == np.array([[3, 0], [3, 0]]))

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

def test_rmul_constant_vector_results():
	x = AutoDiff([3, 1], [2, 1], 2, 1)
	f = 3 * x
	assert np.all(f.val == np.array([[9, 3]]).T)
	assert np.all(f.der == np.array([[6, 0], [3, 0]]))
	assert np.all(f.jacobian == np.array([[3, 0], [3, 0]]))

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

def test_truediv_vector_results():
	x = AutoDiff([3, 1], [2, 1], 2, 1)
	y = AutoDiff([2, -3], [1, 2], 2, 2)
	f = x/y
	assert np.all(f.val == np.array([[3/2, -1/3]]).T)
	assert np.all(f.der == np.array([[1, -0.75], [-1/3, -2/9]]))
	assert np.all(f.jacobian == np.array([[0.5, -0.75], [-1/3, -1/9]]))

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

def test_truediv_constant_vector_results():
	x = AutoDiff([-9, 3], [2, 1], 2, 1)
	f = x / 3
	assert np.all(f.val == np.array([[-3, 1]]).T)
	assert np.all(f.der == np.array([[2/3, 0], [1/3, 0]]))
	assert np.all(f.jacobian == np.array([[1/3, 0], [1/3, 0]]))

# reverse division tests
def test_rtruediv_constant_results():
	# single input case
	# positive numbers
	x = AutoDiff(3, 2)
	f = 6 / x
	assert f.val == 2
	assert f.der == -4/3
	assert f.jacobian == -2/3
	# negative numbers
	x = AutoDiff(-3, 2)
	f = 6 / x
	assert f.val == -2
	assert f.der == -4/3
	assert f.jacobian == -2/3

def test_rtruediv_constant_vector_results():
	x = AutoDiff([-9, 3], [2, 1], 1, 1)
	f = 3 / x
	assert np.all(f.val == np.array([[-1/3, 1]]).T)
	assert np.all(f.der == np.array([[-3*((-9)**(-2))*2], [-3*((3)**(-2))*1]]))
	assert np.all(f.jacobian == np.array([[-3*((-9)**(-2))*1], [-3*((3)**(-2))*1]]))

# power tests
def test_pow_ad_results():
	x = AutoDiff(2, 1)
	f = x**x
	assert f.val == 4
	assert f.der == 4 + np.log(16)
	assert f.jacobian == 4 + np.log(16)

def test_rpow_vector_results():
    x = AutoDiff([4, 3], [2, 1], 2, 1)
    y = AutoDiff([2, 1], [1, 3], 2, 2)
    f = x**y
    assert np.all(f.val == np.array([[4**2, 3**1]]).T)
    assert np.all(f.der == np.array([[2*(4**(2-1))*2, (4**2) * np.log(4) * 1], [1*(3**(1-1))*1, (3**1) * np.log(3)*3]]))
    assert np.all(f.jacobian == np.array([[2*(4**(2-1))*1, (4**2) * np.log(4) * 1], [1*(3**(1-1))*1, (3**1) * np.log(3)*1]]))

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

def test_pow_constant_vector_results():
    x = AutoDiff([4, 3], [2, 1], 1, 1)
    f = x**3
    assert np.all(f.val == np.array([[4**3, 3**3]]).T)
    assert np.all(f.der == np.array([[3*(4**2)*2], [3*(3**2)*1]]))
    assert np.all(f.jacobian == np.array([[3*(4**2)*1], [3*(3**2)*1]]))

# reverse power tests
def test_rpow_constant_results():
	x = AutoDiff(5, 2)
	f = 3**x
	assert f.val == 243
	assert f.der == 486 * np.log(3)
	assert f.jacobian == 243 * np.log(3)

def test_rpow_constant_vector_results():
    x = AutoDiff([4, 3], [2, 1], 1, 1)
    f = 3**x
    assert np.all(f.val == np.array([[3**(4), 3**3]]).T)
    assert np.all(f.der == np.array([[(3**(4))*2 * np.log(3)], [(3**(3))*1 * np.log(3)]]))
    assert np.all(f.jacobian == np.array([[(3**(4))*1 * np.log(3)], [(3**(3))*1 * np.log(3)]]))


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

def test_pos_vector_results():
    x = AutoDiff([4, 3], [2, 1], 1, 1)
    f = + x
    assert np.all(f.val == np.array([[4, 3]]).T)
    assert np.all(f.der == np.array([[2], [1]]))
    assert np.all(f.jacobian == np.array([[1], [1]]))
    y = AutoDiff([-4, -3], [2, 1], 1, 1)
    g = + y
    assert np.all(g.val == np.array([[-4, -3]]).T)
    assert np.all(g.der == np.array([[2], [1]]))
    assert np.all(g.jacobian == np.array([[1], [1]]))

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

def test_neg_vector_results():
    x = AutoDiff([4, 3], [2, 1], 1, 1)
    f = - x
    assert np.all(f.val == np.array([[-4, -3]]).T)
    assert np.all(f.der == np.array([[-2], [-1]]))
    assert np.all(f.jacobian == np.array([[-1], [-1]]))
    y = AutoDiff([-4, -3], [2, 1], 1, 1)
    g = - y
    assert np.all(g.val == np.array([[4, 3]]).T)
    assert np.all(g.der == np.array([[-2], [-1]]))
    assert np.all(g.jacobian == np.array([[-1], [-1]]))

def test_neg_constant_results():
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

def test_abs_vector_results():
    x = AutoDiff([4, 3], [2, 1], 1, 1)
    f = abs(x)
    assert np.all(f.val == np.array([[4, 3]]).T)
    assert np.all(f.der == np.array([[2], [1]]))
    assert np.all(f.jacobian == np.array([[1], [1]]))
    y = AutoDiff([-4, -3], [2, 1], 1, 1)
    g = abs(y)
    assert np.all(g.val == np.array([[4, 3]]).T)
    assert np.all(g.der == np.array([[-2], [-1]]))
    assert np.all(g.jacobian == np.array([[-1], [-1]]))

def test_abs_constant_results():
	x = -3
	f = abs(x)
	assert f == 3

# Comparison tests
def test_eq_results():
	x = AutoDiff(5, 2)
	y = AutoDiff(5, 2)
	z = AutoDiff(5, 1)
	assert x == y
	assert (x == z) == False

def test_eq_vector_results():
    w = AutoDiff([4, 5], [2, 1], 1, 1)
    x = AutoDiff([4, 3], [2, 1], 1, 1)
    y = AutoDiff([4, 3], [2, 1], 1, 1)
    z = AutoDiff([4, 5], [2, 1], 1, 1)
    assert np.all(x == y)
    assert np.all(x==z) == False
    assert np.all(w==x) == False

def test_eq_constant():
	x = AutoDiff(5, 2)
	assert (x == 5) == False

def test_ne_results():
	x = AutoDiff(5, 2)
	y = AutoDiff(5, 2)
	z = AutoDiff(5, 1)
	assert x != z
	assert (x != y) == False

def test_neq_vector_results():
    w = AutoDiff([4, 5], [2, 1], 1, 1)
    x = AutoDiff([4, 3], [2, 1], 1, 1)
    y = AutoDiff([4, 3], [2, 1], 1, 1)
    z = AutoDiff([4, 5], [2, 1], 1, 1)
    assert np.all(x != y) == False
    assert np.all(x!=z)
    assert np.all(w!=x)

def test_ne_constant():
	x = AutoDiff(5, 2)
	assert x != 5

def test_vectorize():
	x = AutoDiff(3, np.array([[2]]), n=3, k=1)
	y = AutoDiff(2, np.array([[2]]), n=3, k=2)
	z = AutoDiff(-1, np.array([[2]]), n=3, k=3)
	fs = [3*x + 2*y + 4*z, x - y + z, x/2, 2*x - 2*y]
	f = vectorize(fs, 3)
	assert np.all(f.val == np.array([[9],[0],[1.5],[2]]))
	assert np.all(f.der == np.array([[6, 4, 8], [2, -2, 2], [1, 0, 0], [4, -4, 0]]))
	assert np.all(f.jacobian == np.array([[3, 2, 4], [1, -1, 1], [0.5, 0, 0], [2, -2, 0]]))
