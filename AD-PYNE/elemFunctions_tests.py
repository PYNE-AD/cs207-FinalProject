import pytest
import numpy as np
from AutoDiff import AutoDiff
import elemFunctions as ef

# hyperbolic arc sine tests

def test_arcsinh_ad_results():
	# positive real numbers
	x = AutoDiff(1, 2)
	sinarchAutoDiff = ef.arcsinh(x)
	assert sinarchAutoDiff.val == np.array([[0.881373587019543]])
	assert sinarchAutoDiff.der == np.array([[2/np.sqrt(2)]])
	assert sinarchAutoDiff.jacobian == np.array([[1/np.sqrt(2)]])
	# negative real numbers
	y = AutoDiff(-1, 2)
	sinarchAutoDiff = ef.arcsinh(y)
	assert sinarchAutoDiff.val == np.array([[-0.881373587019543]])
	assert sinarchAutoDiff.der == np.array([[2/np.sqrt(2)]])
	assert sinarchAutoDiff.jacobian == np.array([[1/np.sqrt(2)]])
	# zero
	z = AutoDiff(0, 2)
	sinarchAutoDiff = ef.arcsinh(z)
	assert sinarchAutoDiff.val == np.array([[0.0]])
	assert sinarchAutoDiff.der == np.array([[2.0]])
	assert sinarchAutoDiff.jacobian == np.array([[1.0]])

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

