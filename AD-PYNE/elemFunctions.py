# Paulina's Portion of elemFunctions.py

import numpy as np
from AutoDiff import AutoDiff

# hyperbolic arc sine
def arcsinh(x):
	''' Compute the hyperbolic arc sine of an AutoDiff object and its derivative.
	
	INPUTS
	======
	x: an AutoDiff object
	
	RETURNS
	=======
	A new AutoDiff object with calculated value and derivative.
	
	EXAMPLES
	========
	>>> x = AutoDiff(0.5, 2, 1)
	>>> myAutoDiff = arcsinh(x)
	>>> myAutoDiff.val
	2.3124383412727525
	>>> myAutoDiff.der
	0.39223227027
	>>> myAutoDiff.jacobian
	0.19611613513818404
	
	'''
	try:
		new_val = np.arcsinh(x.val)
		new_der = ((1)/np.sqrt(x.val**2 + 1))*x.der
		new_jacobian = ((1)/np.sqrt(x.val**2 + 1))*x.jacobian
		return AutoDiff(new_val, new_der, x.n, 0, new_jacobian)
	except AttributeError:
		return np.arcsinh(x)

# hyperbolic arc cosine
def arccosh(x):
	''' Compute the hyperbolic arc cosine of an AutoDiff object and its derivative.
	
	INPUTS
	======
	x: an AutoDiff object
	
	RETURNS
	=======
	A new AutoDiff object with calculated value and derivative.
	
	EXAMPLES
	========
	>>> myAutoDiff = arccosh(x)
	>>> myAutoDiff.val
	np.arccosh(x.val)
	>>> myAutoDiff.der
	(1)/(np.sqrt(x**2 - 1))
	
	'''
	try:
		new_val = np.arccosh(x.val)
		# Derivative of arccosh is only defined when x > 1
		new_der = ((1)/np.sqrt(x.val**2 - 1))*x.der if x.val > 1 else None
		new_jacobian = ((1)/np.sqrt(x.val**2 - 1))*x.new_jacobian if x.val > 1 else None
		return AutoDiff(new_val, new_der, x.n, 0, new_jacobian)
	except AttributeError:
		return np.arccosh(x)

# hyperbolic arc tangent
def arctanh(x):
	''' Compute the hyperbolic arc tangent of an AutoDiff object and its derivative.
	
	INPUTS
	======
	x: an AutoDiff object
	
	RETURNS
	=======
	A new AutoDiff object with calculated value and derivative.
	
	EXAMPLES
	========
	>>> myAutoDiff = arctanh(x)
	>>> myAutoDiff.val
	np.arctanh(x.val)
	>>> myAutoDiff.der
	(1)/(np.sqrt(x**2 - 1))
	
	'''
	try:
		new_val = np.arctanh(x.val)
		new_der = ((1)/np.sqrt(1-x.val**2))*x.der
		new_jacobian = ((1)/np.sqrt(1-x.val**2))*x.jacobian
		return AutoDiff(new_val, new_der, x.n, 0, new_jacobian)
	except AttributeError:
		return np.arctanh(x)

# exponential (e)
def exp(x):
	''' Compute the exponential of an AutoDiff object and its derivative.
	
	INPUTS
	======
	x: an AutoDiff object
	
	RETURNS
	=======
	A new AutoDiff object with calculated value and derivative.
	
	EXAMPLES
	========
	>>> myAutoDiff = exp(x)
	>>> myAutoDiff.val

	>>> myAutoDiff.der
		
	'''
	try:
		new_val = np.exp(x.val)
		new_der = np.exp(x.val) * x.der
		new_jacobian = np.exp(x.val) * x.jacobian
		return AutoDiff(new_val, new_der, x.n, 0, new_jacobian)
	except AttributeError:
		return np.exp(x)

# natural log
def log(x):
	''' Compute the natural log of an AutoDiff object and its derivative.
	
	INPUTS
	======
	x: an AutoDiff object
	
	RETURNS
	=======
	A new AutoDiff object with calculated value and derivative.
	
	EXAMPLES
	========
	>>> myAutoDiff = log(x)
	>>> myAutoDiff.val
	np.log(x.val)
	>>> myAutoDiff.der
	(1)/(np.sqrt(x**2 - 1))
	
	'''
	try:
		new_val = np.log(x.val)
		# Derivative not defined when x = 0
		new_der = (1/x.val)*x.der if x.val != 0 else None
		new_jacobian = (1/x.val)*x.jacobian if x.val != 0 else None
		return AutoDiff(new_val, new_der, x.n, 0, new_jacobian)
	except AttributeError:
		return np.log(x)

# log base 10
def log10(x):
	''' Compute the natural log of an AutoDiff object and its derivative.
	
	INPUTS
	======
	x: an AutoDiff object
	
	RETURNS
	=======
	A new AutoDiff object with calculated value and derivative.
	
	EXAMPLES
	========
	>>> myAutoDiff = log(x)
	>>> myAutoDiff.val
	np.log(x.val)
	>>> myAutoDiff.der
	(1)/(np.sqrt(x**2 - 1))
	
	'''
	try:
		new_val = np.log10(x.val)
		# Derivative not defined when x = 0
		new_der = (1/(x.val*np.log(10)))*x.der if x.val != 0 else None
		new_jacobian = (1/(x.val*np.log(10)))*x.jacobian if x.val != 0 else None
		return AutoDiff(new_val, new_der, x.n, 0, new_jacobian)
	except AttributeError:
		return np.log10(x)

# square root
def sqrt(x):
	''' Compute the square root an AutoDiff object and its derivative.
	
	INPUTS
	======
	x: an AutoDiff object
	
	RETURNS
	=======
	A new AutoDiff object with calculated value and derivative.
	
	EXAMPLES
	========
	>>> x = AutoDiff(np.array([[5]]).T, np.array([[1]]), 1, 1)
	>>> myAutoDiff = sqrt(x)
	>>> myAutoDiff.val
	2.2360679775
	>>> myAutoDiff.der
	0.2236068
		
	'''
	try:
		# Value not defined when x < 0
		new_val = np.sqrt(x.val) if x.val >= 0 else None
		# Derivative not defined when x <= 0
		new_der = 0.5 * x.val ** (-0.5) * x.der if x.val > 0 else None
		new_jacobian = 0.5 * x.val ** (-0.5) * x.jacobian if x.val > 0 else None
		return AutoDiff(new_val, new_der, x.n, 0, new_jacobian)
	except AttributeError:
		return np.sqrt(x)

# Temporary tests

# x = AutoDiff(np.array([[5]]).T, np.array([[1]]), 1, 1)
# print(x.val, x.der, x.jacobian)
# myAutoDiff = sqrt(x)
# print(myAutoDiff.val, myAutoDiff.der, myAutoDiff.jacobian)