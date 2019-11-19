import warnings
import pytest
import numpy as np
from AutoDiff import AutoDiff
import elemFunctions_nikhil as elem

# ------------ARCSIN----------------#

def test_arcsin_ad_results():
	# positive real numbers
	x = AutoDiff(0.5, 2)
	f = elem.arcsin(x)
	assert f.val == np.array([[0.5235987755982988]])
	assert f.der == np.array([[2/np.sqrt(1-0.5**2)]])
	assert f.jacobian == np.array([[1/np.sqrt(1-0.5**2)]])
	
	# out of bounds - undefined sqrt
	with pytest.warns(RuntimeWarning):
		y = AutoDiff(-2, 2)
		f = elem.arcsin(y)
		assert np.isnan(f.val[0][0])
		assert np.isnan(f.der[0][0])
		assert np.isnan(f.jacobian[0][0])

	# out of bounds - div by zero
	with pytest.warns(RuntimeWarning):
		y = AutoDiff(1, 2)
		f = elem.arcsin(y)
		assert f.val == np.array([[1.5707963267948966]])
		assert np.isinf(f.der[0][0])
		assert np.isinf(f.jacobian[0][0])
	
	# zero
	z = AutoDiff(0, 2)
	f = elem.arcsin(z)
	assert f.val == np.array([[0.0]])
	assert f.der == np.array([[2.0]])
	assert f.jacobian == np.array([[1.0]])

def test_arcsin_constant_results():
	a = elem.arcsin(0.7)
	assert a == 0.775397496610753

	b = elem.arcsin(0)
	assert b == 0.0

	with pytest.warns(RuntimeWarning):
		c = elem.arcsin(-5)
		assert np.isnan(c)

def test_arcsin_types():
	with pytest.raises(TypeError):
		elem.arcsin('x')
	with pytest.raises(TypeError):
		elem.arcsin("1234")
	with pytest.raises((TypeError, AttributeError)):
		elem.arcsin({1: 'x'})


# ------------ARCCOS----------------#

def test_arccos_ad_results():
	# positive real numbers
	x = AutoDiff(0.5, 2)
	f = elem.arccos(x)
	assert f.val == np.array([[1.0471975511965976]])
	assert f.der == np.array([[-2/np.sqrt(1-0.5**2)]])
	assert f.jacobian == np.array([[-1/np.sqrt(1-0.5**2)]])
	
	# out of bounds - negative sqrt
	with pytest.warns(RuntimeWarning):
		y = AutoDiff(-2, 2)
		f = elem.arccos(y)
		assert np.isnan(f.val[0][0])
		assert np.isnan(f.der[0][0])
		assert np.isnan(f.jacobian[0][0])
	
	# out of bounds - divide by 0
	with pytest.warns(RuntimeWarning):
		y = AutoDiff(1, 2)
		f = elem.arccos(y)
		assert f.val == np.array([[0]])
		assert np.isneginf(f.der[0][0])
		assert np.isneginf(f.jacobian[0][0])
	
	# zero
	z = AutoDiff(0, 2)
	f = elem.arccos(z)
	assert f.val == np.array([[1.5707963267948966]])
	assert f.der == np.array([[-2.0]])
	assert f.jacobian == np.array([[-1.0]])

def test_arccos_constant_results():
	a = elem.arccos(0.7)
	assert a == 0.7953988301841436
	
	b = elem.arccos(0)
	assert b == 1.5707963267948966

	with pytest.warns(RuntimeWarning):
		c = elem.arccos(-5)
		assert np.isnan(c)
	

def test_arccos_types():
	with pytest.raises(TypeError):
		elem.arccos('x')
	with pytest.raises(TypeError):
		elem.arccos("1234")
	with pytest.raises((TypeError, AttributeError)):
		elem.arccos({1: 'x'})


# ------------ARCTAN----------------#

def test_arctan_ad_results():
	# positive real numbers
	x = AutoDiff(3, 2)
	f = elem.arctan(x)
	assert f.val == np.array([[1.2490457723982544]])
	assert f.der == np.array([[2/(1+(3)**2)]])
	assert f.jacobian == np.array([[1/(1+(3)**2)]])
	# negative real numbers
	y = AutoDiff(-2, 2)
	f = elem.arctan(y)
	assert f.val == np.array([[-1.1071487177940906]])
	assert f.der == np.array([[2/(1+(-2)**2)]])
	assert f.jacobian == np.array([[1/(1+(-2)**2)]])
	# zero
	z = AutoDiff(0, 2)
	f = elem.arctan(z)
	assert f.val == np.array([[0.0]])
	assert f.der == np.array([[2.0]])
	assert f.jacobian == np.array([[1.0]])



def test_arctan_constant_results():
	a = elem.arctan(0.7)
	assert a == 0.6107259643892086
	b = elem.arctan(-5)
	assert b == -1.373400766945016
	c = elem.arctan(0)
	assert c == 0.0

def test_arctan_types():
	with pytest.raises(TypeError):
		elem.arctan('x')
	with pytest.raises(TypeError):
		elem.arctan("1234")
	with pytest.raises((TypeError, AttributeError)):
		elem.arctan({1: 'x'})




# ------------ADD----------------#
def test_add_ad_results():
	# positive real numbers
	x = AutoDiff(3, 2)
	f = x+x
	assert f.val == np.array([[6]])
	assert f.der == np.array([[4]])
	assert f.jacobian == np.array([[2]])
	
	# add with constant
	z = AutoDiff(7, 2)
	f = z+3
	assert f.val == np.array([[10]])
	assert f.der == np.array([[2]])
	assert f.jacobian == np.array([[1]])





# ------------RADD----------------#
def test_radd_ad_results():
	# positive real numbers
	x = AutoDiff(-7, 1)
	f = x+x
	assert f.val == np.array([[-14]])
	assert f.der == np.array([[2]])
	assert f.jacobian == np.array([[2]])
	
	# add with constant
	z = AutoDiff(3, 2)
	f = 7+z
	assert f.val == np.array([[10]])
	assert f.der == np.array([[2]])
	assert f.jacobian == np.array([[1]])
