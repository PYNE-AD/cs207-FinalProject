# Paulina's Portion of elemFunctions.py

import numpy as np
from AutoDiff import AutoDiff
#HYPERBOLIC TRIG FUNCTIONS
def sinh(X):
    ''' Compute the sinh of an AutoDiff object and its derivative.
    INPUTS
    ======
    X: an AutoDiff object
    RETURNS
    =======
    A new AutoDiff object with calculated value and derivative.
    EXAMPLES
    ========
    >>> X = AutoDiff(0.5, 2, 1)
    >>> sinhAutoDiff = sinh(X)
    >>> sinhAutoDiff.val
    0.5210953054937474
    >>> sinhAutoDiff.der
    2.2552519304127614
    >>> sinhAutoDiff.jacobian
    1.1276259652063807
    '''
    try:
        val = np.sinh(X.val)
        der = np.cosh(X.val)*X.der
        jacobian = np.cosh(X.val)*X.jacobian
        return AutoDiff(val, der, X.n, 0, jacobian)
    except AttributeError:
        return np.sinh(X)

def cosh(X):
    ''' Compute the cosh of an AutoDiff object and its derivative.
    INPUTS
    ======
    X: an AutoDiff object
    RETURNS
    =======
    A new AutoDiff object with calculated value and derivative.
    EXAMPLES
    ========
    >>> X = AutoDiff(0.5, 2, 1)
    >>> coshAutoDiff = cosh(X)
    >>> coshAutoDiff.val
    1.1276259652063807
    >>> coshAutoDiff.der
    1.0421906109874948
    >>> coshAutoDiff.jacobian
    0.5210953054937474
    '''
    try:
        val = np.cosh(X.val)
        der = np.sinh(X.val)*X.der
        jacobian = np.sinh(X.val)*X.jacobian
        return AutoDiff(val, der, X.n, 0, jacobian)
    except AttributeError:
        return np.cosh(X)

def tanh(X):
    ''' Compute the tanh of an AutoDiff object and its derivative.
    INPUTS
    ======
    X: an AutoDiff object
    RETURNS
    =======
    A new AutoDiff object with calculated value and derivative.
    EXAMPLES
    ========
    >>> X = AutoDiff(0.5, 2, 1)
    >>> tanhAutoDiff = tanh(X)
    >>> tanhAutoDiff.val
    0.46211715726000974
    >>> tanhAutoDiff.der
    1.572895465931855
    >>>tanhAutoDiff.jacobian
    0.7864477329659275
    '''
    try:
        val = np.tanh(X.val)
        der = 1/(np.cosh(X.val)**2)*X.der
        jacobian = 1/(np.cosh(X.val)**2)*X.jacobian
        return AutoDiff(val, der, X.n, 0, jacobian)
    except AttributeError:
        return np.tanh(X)

# log base 10
def log10(X):
    ''' Compute the natural log of an AutoDiff object and its derivative.
    INPUTS
    ======
    X: an AutoDiff object
    RETURNS
    =======
    A new AutoDiff object with calculated value and derivative.
    EXAMPLES
    ========
    >>> X = AutoDiff(0.5, 2, 1)
    >>> myAutoDiff = log(X)
    >>> myAutoDiff.val
    -0.3010299956639812
    >>> myAutoDiff.der
    1.737177927613007
    >>>myAutoDiff.jacobian
    0.8685889638065035
    '''
    try:
        val = np.log10(X.val)
        # Derivative not defined when X = 0
        der = (1/(X.val*np.log(10)))*X.der if X.val != 0 else None
        jacobian = (1/(X.val*np.log(10)))*X.jacobian if X.val != 0 else None
        return AutoDiff(val, der, X.n, 0, jacobian)
    except AttributeError:
        return np.log10(X)
        
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
		new_der = ((1)/np.sqrt(x.val**2 - 1))*x.der  # if x.val > 1 else None
		new_jacobian = ((1)/np.sqrt(x.val**2 - 1))*x.jacobian  # if x.val > 1 else None
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
		new_der = ((1)/(1-x.val**2))*x.der
		new_jacobian = ((1)/(1-x.val**2))*x.jacobian
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
		new_der = (1/x.val)*x.der # if x.val != 0 else None
		new_jacobian = (1/x.val)*x.jacobian # if x.val != 0 else None
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
		new_der = (1/(x.val*np.log(10)))*x.der # if x.val != 0 else None
		new_jacobian = (1/(x.val*np.log(10)))*x.jacobian # if x.val != 0 else None
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
		new_val = np.sqrt(x.val) # if x.val >= 0 else None
		# Derivative not defined when x <= 0
		new_der = 0.5 * x.val ** (-0.5) * x.der # if x.val > 0 else None
		new_jacobian = 0.5 * x.val ** (-0.5) * x.jacobian # if x.val > 0 else None
		return AutoDiff(new_val, new_der, x.n, 0, new_jacobian)
	except AttributeError:
		return np.sqrt(x)

