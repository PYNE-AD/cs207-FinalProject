import warnings
import pytest
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from AutoDiff import AutoDiff
import elemFunctions as ef

# ------------SINE----------------#

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


# ------------COSINE----------------#

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


# ------------TANGENT----------------#

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
	with pytest.warns(RuntimeWarning):
		h = AutoDiff(np.pi/2, 1.0)
		f = ef.tan(h)
		assert np.isnan(f.val[0][0])
		assert np.isnan(f.der[0][0])
		assert np.isnan(f.jacobian[0][0])

def test_tan_constant_results():
	a = ef.tan(5)
	assert a == -3.380515006246585
	b = ef.tan(-5)
	assert b == 3.380515006246585
	c = ef.tan(0)
	assert c == 0.0
	# Value undefined when cos(val)==0
	with pytest.warns(RuntimeWarning):
		d = ef.tan(np.pi/2)
		assert np.isnan(d)

def test_tan_types():
	with pytest.raises(TypeError):
		ef.tan('x')
	with pytest.raises(TypeError):
		ef.tan("1234")


# ------------ARC SINE----------------#

def test_arcsin_ad_results():
	# positive real numbers
	x = AutoDiff(0.5, 2)
	f = ef.arcsin(x)
	assert f.val == np.array([[0.5235987755982988]])
	assert f.der == np.array([[2/np.sqrt(1-0.5**2)]])
	assert f.jacobian == np.array([[1/np.sqrt(1-0.5**2)]])

	# out of bounds - undefined sqrt
	with pytest.warns(RuntimeWarning):
		y = AutoDiff(-2, 2)
		f = ef.arcsin(y)
		assert np.isnan(f.val[0][0])
		assert np.isnan(f.der[0][0])
		assert np.isnan(f.jacobian[0][0])

	# out of bounds - div by zero
	with pytest.warns(RuntimeWarning):
		y = AutoDiff(1, 2)
		f = ef.arcsin(y)
		assert f.val == np.array([[1.5707963267948966]])
		assert np.isinf(f.der[0][0])
		assert np.isinf(f.jacobian[0][0])

	# zero
	z = AutoDiff(0, 2)
	f = ef.arcsin(z)
	assert f.val == np.array([[0.0]])
	assert f.der == np.array([[2.0]])
	assert f.jacobian == np.array([[1.0]])

def test_arcsin_constant_results():
	a = ef.arcsin(0.7)
	assert a == 0.775397496610753

	b = ef.arcsin(0)
	assert b == 0.0

	with pytest.warns(RuntimeWarning):
		c = ef.arcsin(-5)
		assert np.isnan(c)

def test_arcsin_types():
	with pytest.raises(TypeError):
		ef.arcsin('x')
	with pytest.raises(TypeError):
		ef.arcsin("1234")


# ------------ARC COSINE----------------#

def test_arccos_ad_results():
	# positive real numbers
	x = AutoDiff(0.5, 2)
	f = ef.arccos(x)
	assert f.val == np.array([[1.0471975511965976]])
	assert f.der == np.array([[-2/np.sqrt(1-0.5**2)]])
	assert f.jacobian == np.array([[-1/np.sqrt(1-0.5**2)]])

	# out of bounds - negative sqrt
	with pytest.warns(RuntimeWarning):
		y = AutoDiff(-2, 2)
		f = ef.arccos(y)
		assert np.isnan(f.val[0][0])
		assert np.isnan(f.der[0][0])
		assert np.isnan(f.jacobian[0][0])

	# out of bounds - divide by 0
	with pytest.warns(RuntimeWarning):
		y = AutoDiff(1, 2)
		f = ef.arccos(y)
		assert f.val == np.array([[0]])
		assert np.isneginf(f.der[0][0])
		assert np.isneginf(f.jacobian[0][0])

	# zero
	z = AutoDiff(0, 2)
	f = ef.arccos(z)
	assert f.val == np.array([[1.5707963267948966]])
	assert f.der == np.array([[-2.0]])
	assert f.jacobian == np.array([[-1.0]])

def test_arccos_constant_results():
	a = ef.arccos(0.7)
	assert a == 0.7953988301841436

	b = ef.arccos(0)
	assert b == 1.5707963267948966

	with pytest.warns(RuntimeWarning):
		c = ef.arccos(-5)
		assert np.isnan(c)


def test_arccos_types():
	with pytest.raises(TypeError):
		ef.arccos('x')
	with pytest.raises(TypeError):
		ef.arccos("1234")


# ------------ARC TANGENT----------------#

def test_arctan_ad_results():
	# positive real numbers
	x = AutoDiff(3, 2)
	f = ef.arctan(x)
	assert f.val == np.array([[1.2490457723982544]])
	assert f.der == np.array([[2/(1+(3)**2)]])
	assert f.jacobian == np.array([[1/(1+(3)**2)]])
	# negative real numbers
	y = AutoDiff(-2, 2)
	f = ef.arctan(y)
	assert f.val == np.array([[-1.1071487177940906]])
	assert f.der == np.array([[2/(1+(-2)**2)]])
	assert f.jacobian == np.array([[1/(1+(-2)**2)]])
	# zero
	z = AutoDiff(0, 2)
	f = ef.arctan(z)
	assert f.val == np.array([[0.0]])
	assert f.der == np.array([[2.0]])
	assert f.jacobian == np.array([[1.0]])


def test_arctan_constant_results():
	a = ef.arctan(0.7)
	assert a == 0.6107259643892086
	b = ef.arctan(-5)
	assert b == -1.373400766945016
	c = ef.arctan(0)
	assert c == 0.0

def test_arctan_types():
	with pytest.raises(TypeError):
		ef.arctan('x')
	with pytest.raises(TypeError):
		ef.arctan("1234")



# ------------HYPERBOLIC SINE----------------#
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

# ------------HYPERBOLIC COS----------------#
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

# ------------HYPERBOLIC TAN----------------#
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


# ------------HYPERBOLIC ARC SINE----------------#

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

# ------------HYPERBOLIC ARC COSINE----------------#

def test_arccosh_ad_results():
	# value defined at positive real numbers x >= 1
	# derivative defined at positive real numbers x > 1
	x = AutoDiff(1.1, 2)
	f = ef.arccosh(x)
	assert f.val == 0.4435682543851153
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

# ------------HYPERBOLIC ARC TANGENT----------------#

def test_arctanh_ad_results():
	# value defined at real numbers (-1, 1)
	x = AutoDiff(0.5, 2)
	f = ef.arctanh(x)
	assert f.val == 0.5493061443340549
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

# ---------------EXPONENTIAL----------------#

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

# ---------------LOG----------------#
def test_log_results():
	# value defined at positive real numbers x > 0
	# derivative defined at real numbers x ≠ 0
	X = AutoDiff(0.5, 2)
	f = ef.log(X)
	assert f.val == -0.6931471805599453
	assert f.der == 4
	assert f.jacobian == 2
	# derivative not defined at x = 0
	with pytest.warns(RuntimeWarning):
		Y = AutoDiff(0, 2)
		f = ef.log(Y)
		assert np.isneginf(f.val)
		assert np.isinf(f.der)
		assert np.isinf(f.jacobian)
	# value not defined at x < 0, derivative defined
	with pytest.warns(RuntimeWarning):
		Y = AutoDiff(-0.5, 2)
		f = ef.log(Y)
		assert np.isnan(f.val)
		assert f.der == -4
		assert f.jacobian == -2

def test_log_constant_results():
	a = ef.log(0.5)
	assert a == -0.6931471805599453
	with pytest.warns(RuntimeWarning):
		b = ef.log(0)
		assert np.isneginf(b)
	with pytest.warns(RuntimeWarning):
		b = ef.log(-0.5)
		assert np.isnan(b)

def test_log_types():
	with pytest.raises(TypeError):
		ef.log('x')
	with pytest.raises(TypeError):
		ef.log("1234")


# ---------------LOG10----------------#
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

# ---------------SQUARE ROOT----------------#

def test_sqrt_ad_results():
	# Positive reals
	x = AutoDiff(0.5, 2.0)
	f = ef.sqrt(x)
	assert f.val == np.array([[0.7071067811865476]])
	assert f.der == np.array([[0.5 * 0.5 ** (-0.5) * 2.0]])
	assert f.jacobian == np.array([[0.5 * 0.5 ** (-0.5)]])
	# Value defined but derivative undefined when x == 0
	with pytest.warns(RuntimeWarning):
		y = AutoDiff(0, 2)
		f = ef.sqrt(y)
		assert f.val == np.array([[0]])
		assert np.isinf(f.der[0][0])
		assert np.isinf(f.jacobian[0][0])
	# Value and derivative undefined when x < 0
	with pytest.warns(RuntimeWarning):
		z = AutoDiff(-0.5, 2)
		f = ef.sqrt(z)
		assert np.isnan(f.val[0][0])
		assert np.isnan(f.der[0][0])
		assert np.isnan(f.jacobian[0][0])

def test_sqrt_constant_results():
	a = ef.sqrt(5)
	assert a == 2.23606797749979
	b = ef.sqrt(0)
	assert b == 0.0
	# Value undefined when x < 0
	with pytest.warns(RuntimeWarning):
		c = ef.sqrt(-5)
		assert np.isnan(c)

def test_sqrt_types():
	with pytest.raises(TypeError):
		ef.sqrt('x')
	with pytest.raises(TypeError):
		ef.sqrt("1234")
