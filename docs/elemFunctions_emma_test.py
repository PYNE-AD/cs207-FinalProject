import warnings
import pytest
import numpy as np
from AutoDiff import AutoDiff
import elemFunctions as ef


# Tests for sine function

def test_sin_ad_results():
	# Positive real numbers
	x = AutoDiff(0.5, 2.0)
	f = ef.sin(x)
	assert f.val == np.array([[0.479425538604203]])
	assert f.der == np.array([[np.cos(0.5)*2.0]])
	assert f.jacobian == np.array([[np.cos(0.5)]])
	# Negative real numbers
	y = AutoDiff(-0.5, 2.0)
	f = ef.sin(y)
	assert f.val == np.array([[-0.479425538604203]])
	assert f.der == np.array([[np.cos(-0.5)*2.0]])
	assert f.jacobian == np.array([[np.cos(-0.5)]])
	# Zero
	z = AutoDiff(0.0, 2.0)
	f = ef.sin(z)
	assert f.val == np.array([[0.0]])
	assert f.der == np.array([[2.0]])
	assert f.jacobian == np.array([[1.0]])

def test_sin_constant_results():
	a = ef.sin(5)
	assert a == -0.9589242746631385
	b = ef.sin(-5)
	assert b == 0.9589242746631385
	c = ef.sin(0)
	assert c == 0.0

def test_sin_types():
	with pytest.raises(TypeError):
		ef.sin('x')
	with pytest.raises(TypeError):
		ef.sin("1234")
	with pytest.raises(AttributeError):
		ef.sin({1: 'x'})


# Tests for cosine function

def test_cos_ad_results():
	# Positive real numbers
	x = AutoDiff(0.5, 2.0)
	f = ef.cos(x)
	assert f.val == np.array([[0.8775825618903728]])
	assert f.der == np.array([[np.sin(0.5)*-2.0]])
	assert f.jacobian == np.array([[np.sin(0.5)*-1.0]])
	# Negative real numbers
	y = AutoDiff(-0.5, 2.0)
	f = ef.cos(y)
	assert f.val == np.array([[0.8775825618903728]])
	assert f.der == np.array([[np.sin(-0.5)*-2.0]])
	assert f.jacobian == np.array([[np.sin(-0.5)*-1.0]])
	# Zero
	z = AutoDiff(0.0, 2.0)
	f = ef.cos(z)
	assert f.val == np.array([[1.0]])
	assert f.der == np.array([[0.0]])
	assert f.jacobian == np.array([[0.0]])

def test_cos_constant_results():
	a = ef.cos(5)
	assert a == 0.2836621854632263
	b = ef.cos(-5)
	assert b == 0.2836621854632263
	c = ef.cos(0)
	assert c == 1.0

def test_cos_types():
	with pytest.raises(TypeError):
		ef.cos('x')
	with pytest.raises(TypeError):
		ef.cos("1234")
	with pytest.raises(AttributeError):
		ef.cos({1: 'x'})


# Tests for tangent function

def test_tan_ad_results():
	# Defined value and derivative when cos(val)!=0
	# Positive reals
	x = AutoDiff(0.5, 2.0)
	f = ef.tan(x)
	assert f.val == np.array([[0.5463024898437905]])
	assert f.der == np.array([[2.0 / (np.cos(0.5)**2)]])
	assert f.jacobian == np.array([[1.0 / (np.cos(0.5)**2)]])
	# Negative reals
	y = AutoDiff(-0.5, 2.0)
	f = ef.tan(y)
	assert f.val == np.array([[-0.5463024898437905]])
	assert f.der == np.array([[2.0 / (np.cos(-0.5)**2)]])
	assert f.jacobian == np.array([[1.0 / (np.cos(-0.5)**2)]])
	# Zero
	z = AutoDiff(0.0, 2.0)
	f = ef.tan(z)
	assert f.val == np.array([[0.0]])
	assert f.der == np.array([[2.0]])
	assert f.jacobian == np.array([[1.0]])

	# Undefined value and derivative when cos(val)==0
	with pytest.raises(TypeError):
		h = AutoDiff(np.pi/2, 1.0)
		f = ef.tan(h)

def test_tan_constant_results():
	a = ef.tan(5)
	assert a == -3.380515006246585
	b = ef.tan(-5)
	assert b == 3.380515006246585
	c = ef.tan(0)
	assert c == 0.0
	# Value undefined when cos(val)==0
	with pytest.raises(TypeError):
		ef.tan(np.pi/2)

def test_tan_types():
	with pytest.raises(TypeError):
		ef.tan('x')
	with pytest.raises(TypeError):
		ef.tan("1234")
	with pytest.raises(TypeError):
		ef.tan({1: 'x'})


# Tests for square root function

def test_sqrt_ad_results():
	# Positive reals
	x = AutoDiff(0.5, 2.0)
	f = ef.sqrt(x)
	assert f.val == np.array([[0.7071067811865476]])
	assert f.der == np.array([[0.5 * 0.5 ** (-0.5) * 2.0]])
	assert f.jacobian == np.array([[0.5 * 0.5 ** (-0.5)]])
	# Value defined but derivative undefined when x == 0
	with pytest.raises(TypeError):
		y = AutoDiff(0, 2)
		f = ef.sqrt(y)
	# Value and derivative undefined when x < 0
	with pytest.raises(TypeError):
		z = AutoDiff(-0.5, 2)
		f = ef.sqrt(z)
	print('worked')

def test_sqrt_constant_results():
	a = ef.sqrt(5)
	assert a == 2.23606797749979
	b = ef.sqrt(0)
	assert b == 0.0
	# Value undefined when x < 0
	with pytest.raises(TypeError):
		ef.sqrt(-5)
	print('worked')

def test_sqrt_types():
	with pytest.raises(TypeError):
		ef.sqrt('x')
	with pytest.raises(TypeError):
		ef.sqrt("1234")
	with pytest.raises(TypeError):
		ef.sqrt({1: 'x'})
	print('worked')
