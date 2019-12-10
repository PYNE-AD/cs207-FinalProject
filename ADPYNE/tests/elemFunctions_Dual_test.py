import warnings
import pytest
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from ADPYNE.Dual import Dual
import ADPYNE.elemFunctions as ef
from ADPYNE.Dual import Dual

# ------------SINE----------------#

def test_sin_ad_results():
	# Positive real numbers
	x = Dual(0.5, 2.0)
	f = ef.sin(x)
	assert f.Real == np.array([[np.sin(0.5)]])
	assert f.Dual == np.array([[np.cos(0.5)*2.0]])
	# Negative real numbers
	y = Dual(-0.5, 2.0)
	f = ef.sin(y)
	assert f.Real == np.array([[np.sin(-0.5)]])
	assert f.Dual == np.array([[np.cos(-0.5)*2.0]])
	# Zero
	z = Dual(0.0, 2.0)
	f = ef.sin(z)
	assert f.Real == np.array([[np.sin(0)]])
	assert f.Dual == np.array([[2.0]])

def test_sin_constant_results():
	a = ef.sin(5)
	assert a == np.sin(5)
	b = ef.sin(-5)
	assert b == np.sin(-5)
	c = ef.sin(0)
	assert c == np.sin(0)

def test_sin_types():
	with pytest.raises(TypeError):
		ef.sin('x')
	with pytest.raises(TypeError):
		ef.sin("1234")


# ------------COSINE----------------#

def test_cos_ad_results():
	# Positive real numbers
	x = Dual(0.5, 2.0)
	f = ef.cos(x)
	assert f.Real == np.array([[np.cos(0.5)]])
	assert f.Dual == np.array([[np.sin(0.5)*-2.0]])
	# Negative real numbers
	y = Dual(-0.5, 2.0)
	f = ef.cos(y)
	assert f.Real == np.array([[np.cos(-0.5)]])
	assert f.Dual == np.array([[np.sin(-0.5)*-2.0]])
	# Zero
	z = Dual(0.0, 2.0)
	f = ef.cos(z)
	assert f.Real == np.array([[np.cos(0)]])
	assert f.Dual == np.array([[0.0]])

def test_cos_constant_results():
	a = ef.cos(5)
	assert a == np.cos(5)
	b = ef.cos(-5)
	assert b == np.cos(-5)
	c = ef.cos(0)
	assert c == np.cos(0)

def test_cos_types():
	with pytest.raises(TypeError):
		ef.cos('x')
	with pytest.raises(TypeError):
		ef.cos("1234")


# ------------TANGENT----------------#

def test_tan_ad_results():
	# Defined Realue and Dualivative when cos(Real)!=0
	# Positive reals
	x = Dual(0.5, 2.0)
	f = ef.tan(x)
	assert f.Real == np.array([[np.tan(0.5)]])
	assert f.Dual == np.array([[2.0 / (np.cos(0.5)**2)]])
	# Negative reals
	y = Dual(-0.5, 2.0)
	f = ef.tan(y)
	assert f.Real == np.array([[np.tan(-0.5)]])
	assert f.Dual == np.array([[2.0 / (np.cos(-0.5)**2)]])
	# Zero
	z = Dual(0.0, 2.0)
	f = ef.tan(z)
	assert f.Real == np.array([[np.tan(0)]])
	assert f.Dual == np.array([[2.0]])

	# Undefined Value and Derivative when cos(Real)==0
	with pytest.warns(RuntimeWarning):
		h = Dual(np.pi/2, 1.0)
		f = ef.tan(h)
		print("HERE",f.Real)
		assert np.isnan(f.Real)
		assert np.isnan(f.Dual)

def test_tan_constant_results():
	a = ef.tan(5)
	assert a == np.tan(5)
	b = ef.tan(-5)
	assert b == np.tan(-5)
	c = ef.tan(0)
	assert c == np.tan(0)
	# Realue undefined when cos(Real)==0
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
	x = Dual(0.5, 2)
	f = ef.arcsin(x)
	assert f.Real == np.array([[np.arcsin(0.5)]])
	assert f.Dual == np.array([[2/np.sqrt(1-0.5**2)]])

	# out of bounds - undefined sqrt
	with pytest.warns(RuntimeWarning):
		y = Dual(-2, 2)
		f = ef.arcsin(y)
		assert np.isnan(f.Real)
		assert np.isnan(f.Dual)

	# out of bounds - div by zero
	with pytest.warns(RuntimeWarning):
		y = Dual(1, 2)
		f = ef.arcsin(y)
		assert f.Real == np.array([[np.arcsin(1)]])
		assert np.isinf(f.Dual)

	# zero
	z = Dual(0, 2)
	f = ef.arcsin(z)
	assert f.Real == np.array([[0.0]])
	assert f.Dual == np.array([[2.0]])

def test_arcsin_constant_results():
	a = ef.arcsin(0.7)
	assert a == np.arcsin(0.7)

	b = ef.arcsin(0)
	assert b == np.arcsin(0)

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
	x = Dual(0.5, 2)
	f = ef.arccos(x)
	assert f.Real == np.array([[np.arccos(0.5)]])
	assert f.Dual == np.array([[-2/np.sqrt(1-0.5**2)]])

	# out of bounds - negative sqrt
	with pytest.warns(RuntimeWarning):
		y = Dual(-2, 2)
		f = ef.arccos(y)
		assert np.isnan(f.Real)
		assert np.isnan(f.Dual)

	# out of bounds - divide by 0
	with pytest.warns(RuntimeWarning):
		y = Dual(1, 2)
		f = ef.arccos(y)
		assert f.Real == np.array([[np.arccos(1)]])
		assert np.isneginf(f.Dual)

	# zero
	z = Dual(0, 2)
	f = ef.arccos(z)
	assert f.Real == np.array([[np.arccos(0)]])
	assert f.Dual == np.array([[-2/np.sqrt(1-0**2)]])

def test_arccos_constant_results():
	a = ef.arccos(0.7)
	assert a == np.arccos(0.7)

	b = ef.arccos(0)
	assert b == np.arccos(0)

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
	x = Dual(3, 2)
	f = ef.arctan(x)
	assert f.Real == np.array([[np.arctan(3)]])
	assert f.Dual == np.array([[2/(1+(3)**2)]])

	# negative real numbers
	y = Dual(-2, 2)
	f = ef.arctan(y)
	assert f.Real == np.array([[np.arctan(-2)]])
	assert f.Dual == np.array([[2/(1+(-2)**2)]])

	# zero
	z = Dual(0, 2)
	f = ef.arctan(z)
	assert f.Real == np.array([[np.arctan(0)]])
	assert f.Dual == np.array([[2/(1+(0)**2)]])


def test_arctan_constant_results():
	a = ef.arctan(0.7)
	assert a == np.arctan(0.7)
	b = ef.arctan(-5)
	assert b == np.arctan(-5)
	c = ef.arctan(0)
	assert c == np.arctan(0)

def test_arctan_types():
	with pytest.raises(TypeError):
		ef.arctan('x')
	with pytest.raises(TypeError):
		ef.arctan("1234")



# ------------HYPERBOLIC SINE----------------#
def test_sinh_results():
	X = Dual(0.5, 2)
	f = ef.sinh(X)
	assert f.Real == np.sinh(0.5)
	assert f.Dual == np.array([[2*np.cosh(0.5)]])

def test_sinh_constant_results():
	a = ef.sinh(0.5)
	assert a == np.sinh(0.5)

def test_sinh_types():
	with pytest.raises(TypeError):
		ef.sinh('x')
	with pytest.raises(TypeError):
		ef.sinh("1234")

# ------------HYPERBOLIC COS----------------#
def test_cosh_results():
	X = Dual(0.5, 2)
	f = ef.cosh(X)
	assert f.Real == np.cosh(0.5)
	assert f.Dual == np.array([[np.sinh(0.5)*2]])

def test_cosh_constant_results():
	a = ef.cosh(0.5)
	assert a == np.cosh(0.5)

def test_cosh_types():
	with pytest.raises(TypeError):
		ef.cosh('x')
	with pytest.raises(TypeError):
		ef.cosh("1234")

# ------------HYPERBOLIC TAN----------------#
def test_tanh_results():
	X = Dual(0.5, 2)
	f = ef.tanh(X)
	assert f.Real == np.tanh(0.5)
	assert f.Dual == np.array([[2/(np.cosh(0.5)**2)]])

def test_tanh_constant_results():
	a = ef.tanh(0.5)
	assert a == np.tanh(0.5)

def test_tanh_types():
	with pytest.raises(TypeError):
		ef.tanh('x')
	with pytest.raises(TypeError):
		ef.tanh("1234")


# ------------HYPERBOLIC ARC SINE----------------#

def test_arcsinh_ad_results():
	# positive real numbers
	x = Dual(1, 2)
	f = ef.arcsinh(x)
	assert f.Real == np.arcsinh(1)
	assert f.Dual == np.array([[((2)/np.sqrt((1)**2 + 1))]])
	
	# negative real numbers
	y = Dual(-1, 2)
	f = ef.arcsinh(y)
	assert f.Real == np.arcsinh(-1)
	assert f.Dual == np.array([[((2)/np.sqrt((-1)**2 + 1))]])
	
	# zero
	z = Dual(0, 2)
	f = ef.arcsinh(z)
	assert f.Real == np.arcsinh(0)
	assert f.Dual == np.array([[((2)/np.sqrt((0)**2 + 1))]])

def test_arcsinh_constant_results():
	a = ef.arcsinh(5)
	assert a == np.arcsinh(5)
	b = ef.arcsinh(-5)
	assert b == np.arcsinh(-5)
	c = ef.arcsinh(0)
	assert c == np.arcsinh(0)

def test_arcsinh_types():
	with pytest.raises(TypeError):
		ef.arcsinh('x')
	with pytest.raises(TypeError):
		ef.arcsinh("1234")

# ------------HYPERBOLIC ARC COSINE----------------#

def test_arccosh_ad_results():
	# Realue defined at positive real numbers x >= 1
	# Dualivative defined at positive real numbers x > 1
	x = Dual(1.1, 2)
	f = ef.arccosh(x)
	assert f.Real == np.arccosh(1.1)
	assert f.Dual == np.array([[((2)/np.sqrt((1.1)**2 - 1))]])
	
	# Realue defined at x = 1, Dualivative not defined
	with pytest.warns(RuntimeWarning):
		y = Dual(1, 2)
		f = ef.arccosh(y)
		assert np.isinf(f.Dual)
	
	# neither Realue nor Dualivative defined at x < 1
	with pytest.warns(RuntimeWarning):
		z = Dual(0, 2)
		f = ef.arccosh(z)
		assert np.isnan(f.Real)
		assert np.isnan(f.Dual)

def test_arccosh_constant_results():
	a = ef.arccosh(5)
	assert a == np.arccosh(5)
	# Realue not defined at x < 1
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
	# Realue defined at real numbers (-1, 1)
	x = Dual(0.5, 2)
	f = ef.arctanh(x)
	assert f.Real == np.arctanh(0.5)
	assert f.Dual == np.array([[((2)/(1-(0.5)**2))]])
	
	y = Dual(-0.99999, 2)
	f = ef.arctanh(y)
	assert f.Real == np.arctanh(-0.99999)
	assert f.Dual == np.array([[((2)/(1-(-0.99999)**2))]])

	# test for real numbers not in (-1, 1)
	with pytest.warns(RuntimeWarning):
		z = Dual(-1, 2)
		f = ef.arctanh(z)
		assert np.isinf(f.Real)
		assert np.isinf(f.Dual)

	with pytest.warns(RuntimeWarning):
		z = Dual(10, 2)
		f = ef.arctanh(z)
		assert np.isnan(f.Real)
		assert f.Dual == np.array([[((2)/(1-(10)**2))]])

def test_arctanh_constant_results():
	a = ef.arctanh(0.99999)
	assert a == np.arctanh(0.99999)
	b = ef.arctanh(-0.001)
	assert b == np.arctanh(-0.001)
	with pytest.warns(RuntimeWarning):
		a = ef.arctanh(-1)
		assert np.isinf(a)
	with pytest.warns(RuntimeWarning):
		b = ef.arctanh(10)
		assert np.isnan(b)

def test_arctanh_types():
	with pytest.raises(TypeError):
		ef.arctanh('x')
	with pytest.raises(TypeError):
		ef.arctanh("1234")

# ---------------EXPONENTIAL----------------#

def test_exp_ad_results():
	# Realue defined at all real numbers
	# positive numbers
	x = Dual(10, 2)
	f = ef.exp(x)
	assert f.Real == np.exp(10)
	assert f.Dual == 2*np.exp(10)
	
	y = Dual(-5, 2)
	f = ef.exp(y)
	assert f.Real == np.exp(-5)
	assert f.Dual == 2*np.exp(-5)
	
	z = Dual(0, 2)
	f = ef.exp(z)
	assert f.Real == np.exp(0)
	assert f.Dual == 2*np.exp(0)

def test_exp_constant_results():
	a = ef.exp(0)
	assert a == np.exp(0)
	b = ef.exp(5)
	assert b == np.exp(5)
	c = ef.exp(-10)
	assert c == np.exp(-10)

def test_exp_types():
	with pytest.raises(TypeError):
		ef.exp('x')
	with pytest.raises(TypeError):
		ef.exp("1234")

# ---------------LOG----------------#
def test_log_results():
	# Realue defined at positive real numbers x > 0
	# Dualivative defined at real numbers x ≠ 0
	X = Dual(0.5, 2)
	f = ef.log(X)
	assert f.Real == np.log(0.5)
	assert f.Dual == np.array([[2/0.5]])

	# Dualivative not defined at x = 0
	Y = Dual(0, 2)
	f = ef.log(Y)
	assert np.isneginf(f.Real)
	assert np.isinf(f.Dual)

	# Realue not defined at x < 0, Dualivative defined
	with pytest.warns(RuntimeWarning):
		Y = Dual(-0.5, 2)
		f = ef.log(Y)
		assert np.isnan(f.Real)
		assert f.Dual == np.array([[2/-0.5]])

def test_log_constant_results():
	a = ef.log(0.5)
	assert a == np.log(0.5)
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
	# Realue defined at positive real numbers x > 0
	# Dualivative defined at positive real numbers x > 0
	X = Dual(0.5, 2)
	f = ef.log10(X)
	assert f.Real == np.log10(0.5)
	assert f.Dual == np.array([[(1/((0.5)*np.log(10)))*2]])

	# neither Realue nor Dualivative defined at x = 0
	with pytest.warns(RuntimeWarning):
		Y = Dual(0, 2)
		f = ef.log10(Y)
		assert np.isinf(f.Real)
		assert np.isinf(f.Dual)

	# Realue not defined at x < 0, Dualivative defined
	with pytest.warns(RuntimeWarning):
		Y = Dual(-0.5, 2)
		f = ef.log10(Y)
		assert np.isnan(f.Real)
		assert f.Dual == np.array([[(2/(-0.5*np.log(10)))]])

def test_log10_constant_results():
	a = ef.log10(0.5)
	print(a,"a")
	print(np.log10(0.5),"b")
	assert a == np.log10(0.5)
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

# ---------------LOGBASE----------------#
def test_logbase_results():
	# Realue defined
	# 0<base<1 and x>1
	# base>1 and x>0
	# base=x, base≠0≠1
	# Dualivative defined at positive real numbers x > 0
	X = Dual(0.5, 2)
	f = ef.logbase(X,2)
	assert f.Real == np.log(0.5)/np.log(2)
	assert f.Dual == np.array([[(1/(0.5*np.log(2)))*2]])

	# Realue/Dual not defined at 0<base<1 and x<1
	with pytest.warns(RuntimeWarning):
		Y = Dual(-0.5, 2)
		f = ef.logbase(Y,0.9)
		assert np.isnan(f.Real)
		assert f.Dual == np.array([[(1/(-0.5*np.log(0.9)))*2]])

	# Realue/Dual not defined at base<1 and x>0
	with pytest.warns(RuntimeWarning):
		Y = Dual(1, 2)
		f = ef.logbase(Y,-10)
		assert np.isnan(f.Real)
		assert np.isnan(f.Dual)

	# Realue not defined at base>1 and x<=0
	with pytest.warns(RuntimeWarning):
		Y = Dual(-1, 2)
		f = ef.logbase(Y,10)
		assert np.isnan(f.Real)
		assert f.Dual == np.array([[(1/(-1*np.log(10)))*2]])

	# Realue not defined at x = 0 , where base = 0
	with pytest.warns(RuntimeWarning):
		Y = Dual(0, 2)
		f = ef.logbase(Y,0)
		assert np.isnan(f.Real)
		assert np.isnan(f.Dual)

	# Realue not defined at x = base , where base = 1
	with pytest.warns(RuntimeWarning):
		Y = Dual(1, 2)
		f = ef.logbase(Y,1)
		assert np.isnan(f.Real)
		assert np.isinf(f.Dual)

	# Realue not defined at x = base , where base = 0
	with pytest.warns(RuntimeWarning):
		Y = Dual(0, 2)
		f = ef.logbase(Y,0)
		assert np.isnan(f.Real)
		assert np.isnan(f.Dual)

def test_logbase_constant_results():
	a = ef.logbase(0.5,2)
	assert a == np.log(0.5)/np.log(2)
	with pytest.warns(RuntimeWarning):
		b = ef.logbase(0,2)
		assert np.isneginf(b)
	with pytest.warns(RuntimeWarning):
		b = ef.logbase(-0.5,2)
		assert np.isnan(b)

def test_logbase_types():
	with pytest.raises(TypeError):
		ef.logbase('x',3)
	with pytest.raises(TypeError):
		ef.logbase("1234",3)
	with pytest.raises(TypeError):
		ef.logbase(3,"x")
	with pytest.raises(TypeError):
		ef.logbase(3,"1234")

# ---------------SQUARE ROOT----------------#

def test_sqrt_ad_results():
	# Positive reals
	x = Dual(0.5, 2.0)
	f = ef.sqrt(x)
	assert f.Real == np.array([[np.sqrt(0.5)]])
	assert f.Dual == np.array([[0.5 * 0.5 ** (-0.5) * 2.0]])

	# Realue defined but Dualivative undefined when x == 0
	with pytest.warns(RuntimeWarning):
		y = Dual(0, 2)
		f = ef.sqrt(y)
		assert f.Real == np.array([[0]])
		assert np.isinf(f.Dual)

	# Realue and Dualivative undefined when x < 0
	with pytest.warns(RuntimeWarning):
		z = Dual(-0.5, 2)
		f = ef.sqrt(z)
		assert np.isnan(f.Real)
		assert np.isnan(f.Dual)

def test_sqrt_constant_results():
	a = ef.sqrt(5)
	assert a == np.sqrt(5)
	b = ef.sqrt(0)
	assert b == np.sqrt(0)
	# Realue undefined when x < 0
	with pytest.warns(RuntimeWarning):
		c = ef.sqrt(-5)
		assert np.isnan(c)

def test_sqrt_types():
	with pytest.raises(TypeError):
		ef.sqrt('x')
	with pytest.raises(TypeError):
		ef.sqrt("1234")
