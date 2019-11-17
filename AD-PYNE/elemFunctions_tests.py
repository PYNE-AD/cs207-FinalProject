import warnings
import pytest
import numpy as np
from AutoDiff import AutoDiff
import elemFunctions as ef

# hyperbolic arc sine tests

def test_arcsinh_ad_results():
	# positive real numbers
	x = AutoDiff(1, 2)
	f = ef.arcsinh(x)
	assert f.val == np.array([[0.881373587019543]])
	assert f.der == np.array([[2/np.sqrt(2)]])
	assert f.jacobian == np.array([[1/np.sqrt(2)]])
	# negative real numbers
	y = AutoDiff(-1, 2)
	f = ef.arcsinh(y)
	assert f.val == np.array([[-0.881373587019543]])
	assert f.der == np.array([[2/np.sqrt(2)]])
	assert f.jacobian == np.array([[1/np.sqrt(2)]])
	# zero
	z = AutoDiff(0, 2)
	f = ef.arcsinh(z)
	assert f.val == np.array([[0.0]])
	assert f.der == np.array([[2.0]])
	assert f.jacobian == np.array([[1.0]])

def test_arcsinh_constant_results():
	a = ef.arcsinh(5)
	assert a == 2.3124383412727525
	b = ef.arcsinh(-5)
	assert b == -2.3124383412727525
	c = ef.arcsinh(0)
	assert c == 0.0

def test_arcsinh_types():
	with pytest.raises(TypeError):
		ef.arcsinh('x')
	with pytest.raises(TypeError):
		ef.arcsinh("1234")
	with pytest.raises(AttributeError):
		ef.arcsinh({1: 'x'})

# hyperbolic arc cosine tests

def test_arccosh_ad_results():
	# value defined at positive real numbers x >= 1
	# derivative defined at positive real numbers x > 1
	x = AutoDiff(1.1, 2)
	f = ef.arccosh(x)
	assert f.val == np.array([[0.4435682543851154]])
	assert f.der == np.array([[2/np.sqrt(1.1**2 - 1)]])
	assert f.jacobian == np.array([[1/np.sqrt(1.1**2 - 1)]])
	# value defined at x = 1, derivative not defined
	with pytest.raises(TypeError):
		y = AutoDiff(1, 2)
		f = ef.arccosh(y)
	# neither value nor derivative defined at x < 1
	with pytest.warns(RuntimeWarning):
		with pytest.raises(TypeError):
			z = AutoDiff(0, 2)
			f = ef.arccosh(z)

def test_arccosh_constant_results():
	a = ef.arccosh(5)
	assert a == 2.2924316695611777
	# value not defined at x < 1
	with pytest.warns(RuntimeWarning):
		ef.arccosh(0.9)

def test_arccosh_types():
	with pytest.raises(TypeError):
		ef.arccosh('x')
	with pytest.raises(TypeError):
		ef.arccosh("1234")
	with pytest.raises(AttributeError):
		ef.arccosh({1: 'x'})

# hyperbolic arc tangent tests

def test_arctanh_ad_results():
	# value defined at real numbers (-1, 1)
	x = AutoDiff(0.5, 2)
	f = ef.arctanh(x)
	assert f.val == np.array([[0.5493061443340548]])
	assert f.der == np.array([[2/(1-(0.5)**2)]])
	assert f.jacobian == np.array([[1/(1-(0.5)**2)]])
	y = AutoDiff(-0.99999, 2)
	f = ef.arctanh(y)
	assert f.val == np.array([[-6.1030338227611125]])
	assert f.der == np.array([[2/(1-(-0.99999)**2)]])
	assert f.jacobian == np.array([[1/(1-(-0.99999)**2)]])
	# test for real numbers not in (-1, 1)
	with pytest.warns(RuntimeWarning):
		z = AutoDiff(-1, 2)
		f = ef.arctanh(z)
	with pytest.warns(RuntimeWarning):
		z = AutoDiff(10, 2)
		f = ef.arctanh(z)

def test_arctanh_constant_results():
	a = ef.arctanh(0.99999)
	assert a == 6.1030338227611125
	b = ef.arctanh(-0.001)
	assert b == -0.0010000003333335333
	with pytest.warns(RuntimeWarning):
		ef.arctanh(1)
	with pytest.warns(RuntimeWarning):
		ef.arctanh(-10)

def test_arctanh_types():
	with pytest.raises(TypeError):
		ef.arctanh('x')
	with pytest.raises(TypeError):
		ef.arctanh("1234")
	with pytest.raises(AttributeError):
		ef.arctanh({1: 'x'})

# exponential

def test_exp_ad_results():
	# value defined at all real numbers
	# positive numbers
	x = AutoDiff(10, 2)
	f = ef.exp(x)
	assert f.val == np.array([[22026.465794806718]])
	assert f.der == np.array([[2*22026.465794806718]])
	assert f.jacobian == np.array([[22026.465794806718]])
	y = AutoDiff(-5, 2)
	f = ef.exp(y)
	assert f.val == np.array([[0.006737946999085467]])
	assert f.der == np.array([[2*0.006737946999085467]])
	assert f.jacobian == np.array([[0.006737946999085467]])
	z = AutoDiff(0, 2)
	f = ef.exp(z)
	assert f.val == np.array([[1.0]])
	assert f.der == np.array([[2.0]])
	assert f.jacobian == np.array([[1.0]])

def test_exp_constant_results():
	a = ef.exp(0)
	assert a == 1
	b = ef.exp(5)
	assert b == 148.4131591025766	
	c = ef.exp(-10)
	assert c == 4.5399929762484854e-05	

def test_exp_types():
	with pytest.raises(TypeError):
		ef.exp('x')
	with pytest.raises(TypeError):
		ef.exp("1234")
	with pytest.raises(AttributeError):
		ef.exp({1: 'x'})
	