import warnings
import pytest
import numpy as np
from .AutoDiff import AutoDiff
import elemFunctions as ef

# hyperbolic sine tests
def test_sinh_results():
	X = AutoDiff(0.5, 2)
	f = ef.sinh(X)   
	assert f.val == 0.5210953054937474
	assert f.der == 2.2552519304127614
	assert f.jacobian == 1.1276259652063807

def test_sinh_constant_results():
	a = ef.sinh(0.5)
	assert a == 0.5210953054937474

def test_sinh_types():
	with pytest.raises(TypeError):
		ef.sinh('x')
	with pytest.raises(TypeError):
		ef.sinh("1234")
	with pytest.raises(TypeError, AttributeError):
		ef.sinh({1: 'x'})        

# hyperbolic cos tests
def test_cosh_results():
	X = AutoDiff(0.5, 2)
	f = ef.cosh(X)    
	assert f.val == 1.1276259652063807
	assert f.der == 1.0421906109874948
	assert f.jacobian == 0.5210953054937474
	
def test_cosh_constant_results():
	a = ef.cosh(0.5)
	assert a == 1.1276259652063807 

def test_cosh_types():
	with pytest.raises(TypeError):
		ef.cosh('x')
	with pytest.raises(TypeError):
		ef.cosh("1234")
	with pytest.raises(TypeError, AttributeError):
		ef.cosh({1: 'x'})       
		
# hyperbolic tan tests
def test_tanh_results():
	X = AutoDiff(0.5, 2)
	f = ef.tanh(X)    
	assert f.val == 0.46211715726000974
	assert f.der == 1.572895465931855
	assert f.jacobian == 0.7864477329659275
	
def test_tanh_constant_results():
	a = ef.tanh(0.5)
	assert a == 0.46211715726000974 

def test_tanh_types():
	with pytest.raises(TypeError):
		ef.tanh('x')
	with pytest.raises(TypeError):
		ef.tanh("1234")
	with pytest.raises(TypeError, AttributeError):
		ef.tanh({1: 'x'})         

# log10 test
def test_log10_results():
	# value defined at positive real numbers x > 0
	# derivative defined at positive real numbers x > 0
	X = AutoDiff(0.5, 2)
	f = ef.log10(X)    
	assert f.val == -0.3010299956639812
	assert f.der == 1.737177927613007
	assert f.jacobian == 0.8685889638065035
	# neither value nor derivative defined at x = 0
	with pytest.warns(RuntimeWarning):
		Y = AutoDiff(0, 2)
		f = ef.log10(Y)
		assert np.isinf(f.val)
		assert np.isinf(f.der)
		assert np.isinf(f.jacobian)
	# value not defined at x < 0, derivative defined
	with pytest.warns(RuntimeWarning):
		Y = AutoDiff(-0.5, 2)
		f = ef.log10(Y)
		assert np.isnan(f.val)
		assert f.der == (2/(-0.5*np.log(10)))
		assert f.jacobian == 1/(-0.5*np.log(10))
	
def test_log10_constant_results():
	a = ef.log10(0.5)
	assert a == -0.3010299956639812
	with pytest.warns(RuntimeWarning):
		b = ef.log10(0)
		assert np.isinf(b)
	with pytest.warns(RuntimeWarning):
		b = ef.log10(-0.5)
		assert np.isnan(b)	

def test_log10_types():
	with pytest.raises(TypeError):
		ef.log10('x')
	with pytest.raises(TypeError):
		ef.log10("1234")
	with pytest.raises(TypeError, AttributeError):
		ef.log10({1: 'x'})  

# hyperbolic arc sine tests

def test_arcsinh_ad_results():
	# positive real numbers
	x = AutoDiff(1, 2)
	f = ef.arcsinh(x)
	assert f.val == 0.881373587019543
	assert f.der == 2/np.sqrt(2)
	assert f.jacobian == 1/np.sqrt(2)
	# negative real numbers
	y = AutoDiff(-1, 2)
	f = ef.arcsinh(y)
	assert f.val == -0.881373587019543
	assert f.der == 2/np.sqrt(2)
	assert f.jacobian == 1/np.sqrt(2)
	# zero
	z = AutoDiff(0, 2)
	f = ef.arcsinh(z)
	assert f.val == 0.0
	assert f.der == 2.0
	assert f.jacobian == 1.0

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
	with pytest.raises(TypeError, AttributeError):
		ef.arcsinh({1: 'x'})

# hyperbolic arc cosine tests

def test_arccosh_ad_results():
	# value defined at positive real numbers x >= 1
	# derivative defined at positive real numbers x > 1
	x = AutoDiff(1.1, 2)
	f = ef.arccosh(x)
	assert f.val == 0.4435682543851154
	assert f.der == 2/np.sqrt(1.1**2 - 1)
	assert f.jacobian == 1/np.sqrt(1.1**2 - 1)
	# value defined at x = 1, derivative not defined
	with pytest.warns(RuntimeWarning):
		y = AutoDiff(1, 2)
		f = ef.arccosh(y)
		assert np.isinf(f.der)
		assert np.isinf(f.jacobian)
	# neither value nor derivative defined at x < 1
	with pytest.warns(RuntimeWarning):
		z = AutoDiff(0, 2)
		f = ef.arccosh(z)
		assert np.isnan(f.val)
		assert np.isnan(f.der)
		assert np.isnan(f.jacobian)

def test_arccosh_constant_results():
	a = ef.arccosh(5)
	assert a == 2.2924316695611777
	# value not defined at x < 1
	with pytest.warns(RuntimeWarning):
		a = ef.arccosh(0.9)
		assert np.isnan(a)

def test_arccosh_types():
	with pytest.raises(TypeError):
		ef.arccosh('x')
	with pytest.raises(TypeError):
		ef.arccosh("1234")
	with pytest.raises(TypeError, AttributeError):
		ef.arccosh({1: 'x'})

# hyperbolic arc tangent tests

def test_arctanh_ad_results():
	# value defined at real numbers (-1, 1)
	x = AutoDiff(0.5, 2)
	f = ef.arctanh(x)
	assert f.val == 0.5493061443340548
	assert f.der == 2/(1-(0.5)**2)
	assert f.jacobian == 1/(1-(0.5)**2)
	y = AutoDiff(-0.99999, 2)
	f = ef.arctanh(y)
	assert f.val == -6.1030338227611125
	assert f.der == 2/(1-(-0.99999)**2)
	assert f.jacobian == 1/(1-(-0.99999)**2)
	# test for real numbers not in (-1, 1)
	with pytest.warns(RuntimeWarning):
		z = AutoDiff(-1, 2)
		f = ef.arctanh(z)
		assert np.isinf(f.val)
		assert np.isinf(f.der)
		assert np.isinf(f.jacobian)
	with pytest.warns(RuntimeWarning):
		z = AutoDiff(10, 2)
		f = ef.arctanh(z)
		assert np.isnan(f.val)
		assert f.der == ((2)/(1-10**2))
		assert f.jacobian == ((1)/(1-10**2))

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
	with pytest.raises(TypeError, AttributeError):
		ef.arctanh({1: 'x'})

# exponential

def test_exp_ad_results():
	# value defined at all real numbers
	# positive numbers
	x = AutoDiff(10, 2)
	f = ef.exp(x)
	assert f.val == 22026.465794806718
	assert f.der == 2*22026.465794806718
	assert f.jacobian == 22026.465794806718
	y = AutoDiff(-5, 2)
	f = ef.exp(y)
	assert f.val == 0.006737946999085467
	assert f.der == 2*0.006737946999085467
	assert f.jacobian == 0.006737946999085467
	z = AutoDiff(0, 2)
	f = ef.exp(z)
	assert f.val == 1.0
	assert f.der == 2.0
	assert f.jacobian == 1.0

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
	with pytest.raises(TypeError, AttributeError):
		ef.exp({1: 'x'})
	
print(1/2)