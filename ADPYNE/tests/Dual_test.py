import warnings
import pytest
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from ADPYNE.Dual import Dual, makeHessianVars
import ADPYNE.elemFunctions as ef
import ADPYNE.elemFunctions as ef
from ADPYNE.Hessian import Hessian

# higher order derivative and __str__ test
def test_higherorderprint():
	x = Dual(2, 1)
	assert str(x) == "2 + 1ε"
	y = x.makeHighestOrder(3)
	f = y
	f.buildCoefficients(3)
	assert f.coefficients == [2.0, 1.0, 0.0, -0.0]
	assert str(f) == "[2.0, 1.0, 0.0, -0.0]"

# addition tests
def test_add_dual_results():
	# single input cases
	# positive numbers
	x = Dual(5, 1)
	f = x + x
	assert f.Real == 10
	assert f.Dual == 2
	# negative numbers
	y = Dual(-5, 1)
	f = y + y
	assert f.Real == -10
	assert f.Dual == 2

def test_add_vector_results():
	x = Dual(np.array([[3],[1]]), np.array([[2, 1]]).T)
	y = Dual(np.array([[2],[-3]]), np.array([[3, 2]]).T)
	f = x + y
	assert np.all(f.Real == np.array([[5], [-2]]))
	assert np.all(f.Dual == np.array([[5], [3]]))

def test_add_constant_results():
	# single input case
	# positive numbers
	x = Dual(5, 1)
	f = x + 3
	assert f.Real == 8
	assert f.Dual == 1
	# negative numbers
	x = Dual(-5, 1)
	f = x + 3
	assert f.Real == -2
	assert f.Dual == 1

def test_add_constant_vector_results():
	x = Dual(np.array([[1, 3]]).T, np.array([[2, 1]]).T)
	f = x + 3
	assert np.all(f.Real == np.array([[4, 6]]).T)
	assert np.all(f.Dual == np.array([[2], [1]]))

# reverse addition tests
def test_radd_constant_results():
	# single input case
	# positive numbers
	x = Dual(5, 1)
	f = 3 + x
	assert f.Real == 8
	assert f.Dual == 1
	# negative numbers
	x = Dual(-5, 1)
	f = 3 + x
	assert f.Real == -2
	assert f.Dual == 1

def test_radd_constant_vector_results():
	x = Dual(np.array([[1, 3]]).T, np.array([[2, 1]]).T)
	f = 3 + x
	assert np.all(f.Real == np.array([[4, 6]]).T)
	assert np.all(f.Dual == np.array([[2], [1]]))

# subtraction tests
def test_sub_ad_results():
	# single input cases
	# positive numbers
	x = Dual(5, 2)
	f = x - x
	assert f.Real == 0
	assert f.Dual == 0
	# negative numbers
	y = Dual(-5, 2)
	f = y - y
	assert f.Real == 0
	assert f.Dual == 0

def test_sub_vector_results():
	x = Dual(np.array([[3, 1]]), np.array([[2, 1]]))
	y = Dual(np.array([[2, -3]]), np.array([[3, 2]]))
	f = x - y
	assert np.all(f.Real == np.array([[1, 4]]))
	assert np.all(f.Dual == np.array([[-1], [-1]]))

def test_sub_constant_results():
	# single input case
	# positive numbers
	x = Dual(5, 2)
	f = x - 3
	assert f.Real == 2
	assert f.Dual == 2
	# negative numbers
	x = Dual(-5, 2)
	f = x - 3
	assert f.Real == -8
	assert f.Dual == 2

def test_sub_constant_vector_results():
	x = Dual(np.array([[1, 3]]), np.array([[2, 1]]))
	f = x - 3
	assert np.all(f.Real == np.array([[-2, 0]]))
	assert np.all(f.Dual == np.array([[2, 1]]))

# reverse subtraction tests
def test_rsub_constant_results():
	# single input case
	# positive numbers
	x = Dual(5, 2)
	f = 3 - x
	assert f.Real == -2
	assert f.Dual == -2
	# negative numbers
	x = Dual(-5, 2)
	f = 3 - x
	assert f.Real == 8
	assert f.Dual == -2

def test_rsub_constant_vector_results():
	x = Dual(np.array([[1, 3]]), np.array([[2, 1]]))
	f = 3 - x
	assert np.all(f.Real == np.array([[2, 0]]))
	assert np.all(f.Dual == np.array([[-2, -1]]))

# multiplication tests
def test_mul_dual_results():
	# single input case
	# positive numbers
	x = Dual(5, 2)
	f = x * x
	assert f.Real == 25
	assert f.Dual == 20
	# negative numbers
	x = Dual(-5, 2)
	f = x * x
	assert f.Real == 25
	assert f.Dual == -20

def test_mul_vector_results():
	x = Dual(np.array([[3, 1]]), np.array([[2, 1]]))
	y = Dual(np.array([[2, -1]]), np.array([[1, 2]]))
	f = x*y
	assert np.all(f.Real == np.array([[6, -1]]))
	assert np.all(f.Dual == np.array([[7, 1]]))

def test_mul_constant_results():
	# single input case
	# positive numbers
	x = Dual(5, 2)
	f = x * 3
	assert f.Real == 15
	assert f.Dual == 6
	# negative numbers
	x = Dual(-5, 2)
	f = x * 3
	assert f.Real == -15
	assert f.Dual == 6

def test_mul_constant_vector_results():
	x = Dual(np.array([[3, 1]]), np.array([[2, 1]]))
	f = x * 3
	assert np.all(f.Real == np.array([[9, 3]]))
	assert np.all(f.Dual == np.array([[6, 3]]))

# reverse multiplication tests
def test_rmul_constant_results():
	# single input case
	# positive numbers
	x = Dual(5, 2)
	f = 3 * x
	assert f.Real == 15
	assert f.Dual == 6
	# negative numbers
	x = Dual(-5, 2)
	f = 3 * x
	assert f.Real == -15
	assert f.Dual == 6

def test_rmul_constant_vector_results():
	x = Dual(np.array([[3, 1]]), np.array([[2, 1]]))
	f = 3 * x
	assert np.all(f.Real == np.array([[9, 3]]))
	assert np.all(f.Dual == np.array([[6, 3]]))

# division tests
def test_truediv_dual_results():
	# single input case
	# positive numbers
	x = Dual(5, 2)
	f = x / x
	assert f.Real == 1
	assert f.Dual == 0
	# negative numbers
	x = Dual(-5, 2)
	f = x / x
	assert f.Real == 1
	assert f.Dual == 0

def test_truediv_vector_results():
	x = Dual(np.array([[3, 1]]), np.array([[2, 1]]))
	y = Dual(np.array([[2, -3]]), np.array([[1, 2]]))
	f = x/y
	assert np.all(f.Real == np.array([[3/2, -1/3]]))
	assert np.all(f.Dual == np.array([[1/4, -5/9]]))

def test_truediv_constant_results():
	# single input case
	# positive numbers
	x = Dual(9, 6)
	f = x / 3
	assert f.Real == 3
	assert f.Dual == 2
	# negative numbers
	x = Dual(-9, 6)
	f = x / 3
	assert f.Real == -3
	assert f.Dual == 2

def test_truediv_constant_vector_results():
	x = Dual(np.array([[9, 3]]), np.array([[2, 1]]))
	f = x / 3
	assert np.all(f.Real == np.array([[3, 1]]))
	assert np.all(f.Dual == np.array([[2/3, 1/3]]))

# reverse division tests
def test_rtruediv_constant_results():
	# single input case
	# positive numbers
	x = Dual(3, 2)
	f = 6 / x
	assert f.Real == 2
	assert f.Dual == -4/3
	# negative numbers
	x = Dual(-3, 2)
	f = 6 / x
	assert f.Real == -2
	assert f.Dual == -4/3

def test_rtruediv_constant_vector_results():
	x = Dual(np.array([[-9, 1]]), np.array([[2, 1]]))
	f = 3 / x
	assert np.all(f.Real == np.array([[-1/3, 3]]))
	assert np.all(f.Dual == np.array([[-2/27, -3]]))

# power tests
def test_pow_dual_results():
	x = Dual(2, 1)
	f = x**x
	assert f.Real == 4
	assert f.Dual == 1

def test_rpow_vector_results():
	x = Dual(np.array([[4, 3]]), np.array([[2, 1]]))
	y = Dual(np.array([[2, 1]]), np.array([[1, 3]]))
	f = x**y
	assert np.all(f.Real == np.array([[16, 3]]))
	assert np.all(f.Dual == np.array([[2, 1]]))

def test_pow_constant_results():
	# positive numbers
	x = Dual(5, 2)
	f = x**3
	assert f.Real == 125
	assert f.Dual == 150
	# negative numbers
	x = Dual(-5, 2)
	f = x**3
	assert f.Real == -125
	assert f.Dual == 150

def test_pow_constant_vector_results():
	x = Dual(np.array([[4, 3]]), np.array([[2, 1]]))
	f = x**3
	assert np.all(f.Real == np.array([[64, 27]]))
	assert np.all(f.Dual == np.array([[96, 27]]))

# reverse power tests
def test_rpow_constant_results():
	x = Dual(5, 2)
	f = 3**x
	assert f.Real == 243
	assert f.Dual == 486 * np.log(3)
test_rpow_constant_results()

def test_rpow_constant_vector_results():
	x = Dual(np.array([[4, 3]]), np.array([[2, 1]]))
	f = 3**x
	assert np.all(f.Real == np.array([[81, 27]]))
	assert np.all(f.Dual == np.array([[162*np.log(3), 27*np.log(3)]]))
test_rpow_constant_vector_results()

# positive tests
def test_pos_results():
	# positive numbers
	x = Dual(5, 2)
	f = + x
	assert f.Real == 5
	assert f.Dual == 2
	# negative numbers
	y = Dual(-5, 2)
	f = + y
	assert f.Real == -5
	assert f.Dual == 2

def test_pos_vector_results():
	x = Dual(np.array([[4, 3]]), np.array([[2, 1]]))
	f = + x
	assert np.all(f.Real == np.array([[4, 3]]))
	assert np.all(f.Dual == np.array([[2, 1]]))
	y = Dual(np.array([[-4, -3]]), np.array([[2, 1]]))
	g = + y
	assert np.all(g.Real == np.array([[-4, -3]]))
	assert np.all(g.Dual == np.array([[2, 1]]))

# negation tests
def test_neg_results():
	# positive numbers
	x = Dual(5, 2)
	f = - x
	assert f.Real == -5
	assert f.Dual == -2
	# negative numbers
	y = Dual(-5, 2)
	f = - y
	assert f.Real == 5
	assert f.Dual == -2

def test_neg_vector_results():
	x = Dual(np.array([[4, 3]]), np.array([[2, 1]]))
	f = - x
	assert np.all(f.Real == np.array([[-4, -3]]))
	assert np.all(f.Dual == np.array([[-2, -1]]))
	y = Dual(np.array([[-4, -3]]), np.array([[2, 1]]))
	g = - y
	assert np.all(g.Real == np.array([[4, 3]]))
	assert np.all(g.Dual == np.array([[-2, -1]]))

def test_neg_constant_results():
	x = 3
	f = - x
	assert f == -3

# absolute value tests
def test_abs_results():
	# positive numbers
	x = Dual(5, 2)
	f = abs(x)
	assert f.Real == 5
	assert f.Dual == 0.4
	# negative numbers
	y = Dual(-5, 2)
	f = abs(y)
	assert f.Real == 5
	assert f.Dual == 0.4

def test_abs_vector_results():
	x = Dual(np.array([[4, 3]]), np.array([[2, 1]]))
	f = abs(x)
	assert np.all(f.Real == np.array([[4, 3]]))
	assert np.all(f.Dual == np.array([[1/2, 1/3]]))
	y = Dual(np.array([[-4, -3]]), np.array([[2, 1]]))
	g = abs(y)
	assert np.all(g.Real == np.array([[4, 3]]))
	assert np.all(g.Dual == np.array([[1/2, 1/3]]))

def test_abs_constant_results():
	x = -3
	f = abs(x)
	assert f == 3
	y = 3
	f = abs(y)
	assert f == 3

# Comparison tests
def test_eq_results():
	x = Dual(5, 2)
	y = Dual(5, 2)
	z = Dual(5, 1)
	assert x == y
	assert (x == z) == False

def test_eq_vector_results():
	w = Dual(np.array([[4, 5]]), np.array([[2, 1]]))
	x = Dual(np.array([[4, 3]]), np.array([[2, 1]]))
	y = Dual(np.array([[4, 3]]), np.array([[2, 1]]))
	z = Dual(np.array([[4, 5]]), np.array([[2, 1]]))
	assert np.all(x != y) == False
	assert np.all(x==z) == False
	assert np.all(w==x) == False

def test_eq_constant():
	x = Dual(5, 2)
	assert (x == 5) == False

def test_ne_results():
	x = Dual(5, 2)
	y = Dual(5, 2)
	z = Dual(5, 1)
	assert x != z
	assert (x != y) == False

def test_neq_vector_results():
	w = Dual(np.array([[4, 5]]), np.array([[2, 1]]))
	x = Dual(np.array([[4, 3]]), np.array([[2, 1]]))
	y = Dual(np.array([[4, 3]]), np.array([[2, 1]]))
	z = Dual(np.array([[4, 5]]), np.array([[2, 1]]))
	assert np.all(x != y) == False
	assert np.all(x!=z)
	assert np.all(w!=x)

def test_ne_constant():
	x = Dual(5, 2)
	assert x != 5


def test_hessian():
	x = Dual(3, np.array([1,0]))
	y = Dual(1, np.array([0,1]))
	xh,yh = makeHessianVars(x,y)
	func = (xh**2)*(ef.exp(yh))
	f = Hessian(func)
	assert f.value == (3**2)*np.exp(1)
	assert np.all(f.firstDer == np.array([2*3*np.exp(1), (3**2)*np.exp(1)]))
	assert np.all(f.hessian == np.array([[2*np.exp(1), 2*3*np.exp(1)],[2*3*np.exp(1), (3**2)*np.exp(1)]]))


def test_hessian2():
	x = Dual(0.5, np.array([1,0]))
	y = Dual(0.2, np.array([0,1]))
	xh,yh = makeHessianVars(x,y)
	func = ef.arcsin(xh)*ef.log(yh)**(-1)
	f = Hessian(func)
	assert f.value == np.arcsin(0.5)/np.log(0.2)
	fder = np.array([ 1/(np.sqrt(1-0.5**2)*np.log(0.2)), -np.arcsin(0.5)/(0.2*(np.log(0.2)**2)) ])
	assert np.allclose(f.firstDer,fder)
	fhess = np.array([[0.5/((1-0.5**2)**(3/2)*np.log(0.2)), -1/(np.sqrt(1-0.5**2)*0.2*(np.log(0.2)**2))],[-1/(np.sqrt(1-0.5**2)*0.2*(np.log(0.2)**2)), 2*np.arcsin(0.5)/((0.2**2)*(np.log(0.2)**3))+ np.arcsin(0.5)/((0.2**2)*(np.log(0.2)**2))]])
	assert np.allclose(f.hessian,fhess)