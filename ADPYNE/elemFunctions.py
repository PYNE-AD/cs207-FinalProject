import warnings
import numpy as np
from AutoDiff import AutoDiff


#-------------------BASE TRIG FUNCTIONS-------------------#
# Sine function
def sin(x):
	''' Compute the sine of an AutoDiff object and its derivative.

	INPUTS
	======
	x: an AutoDiff object

	RETURNS
	=======
	A new AutoDiff object with calculated value and derivative.

	EXAMPLES
	========
	>>> x = AutoDiff(0.5, 2.0, 1.0)
	>>> myAutoDiff = sin(x)
	>>> myAutoDiff.val
	0.479425538604
	>>> myAutoDiff.der
	1.75516512378
	>>> myAutoDiff.jacobian
	0.87758256189

	'''
	try:
		new_val = np.sin(x.val)
		new_der = np.cos(x.val) * x.der
		new_jacobian = np.cos(x.val) * x.jacobian
		return AutoDiff(new_val, new_der, x.n, 0, new_jacobian)
	except AttributeError:
		return np.sin(x)

# Cosine function
def cos(x):
	''' Compute the cosine of an AutoDiff object and its derivative.

	INPUTS
	======
	x: an AutoDiff object

	RETURNS
	=======
	A new AutoDiff object with calculated value and derivative.

	EXAMPLES
	========
	>>> x = AutoDiff(0.5, 2.0, 1.0)
	>>> myAutoDiff = cos(x)
	>>> myAutoDiff.val
	0.87758256189
	>>> myAutoDiff.der
	-0.958851077208
	>>> myAutoDiff.jacobian
	-0.479425538604

	'''
	try:
		new_val = np.cos(x.val)
		new_der = -1.0 * np.sin(x.val) * x.der
		new_jacobian = -1.0 * np.sin(x.val) * x.jacobian
		return AutoDiff(new_val, new_der, x.n, 0, new_jacobian)
	except AttributeError:
		return np.cos(x)

# Tangent function
def tan(x):
	''' Compute the tangent of an AutoDiff object and its derivative.

	INPUTS
	======
	x: an AutoDiff object

	RETURNS
	=======
	A new AutoDiff object with calculated value and derivative.

	EXAMPLES
	========
	>>> x = AutoDiff(0.5, 2.0, 1.0)
	>>> myAutoDiff = tan(x)
	>>> myAutoDiff.val
	0.546302489844
	>>> myAutoDiff.der
	2.59689282082
	>>> myAutoDiff.jacobian
	1.29844641041

	'''
	try:
		# Value and derivative undefined when divisible by pi/2 but not pi
		# To make sure the asymptotes are undefined:
		if x.val%(np.pi/2)==0 and x.val%np.pi!=0:
			new_val = np.nan
			new_der = np.nan
			new_jacobian = np.nan
			warnings.warn('Undefined at value', RuntimeWarning)
		else:
			new_val = np.tan(x.val)
			new_der = x.der / (np.cos(x.val)**2.0)
			new_jacobian = x.jacobian / (np.cos(x.val)**2.0)
		return AutoDiff(new_val, new_der, x.n, 0, new_jacobian)
	except AttributeError:
		if x%(np.pi/2)==0 and x%np.pi!=0:
			warnings.warn('Undefined at value', RuntimeWarning)
			return np.nan
		else:
			return np.tan(x)



#-------------------INVERSE TRIG FUNCTIONS-------------------#
# arc sin
def arcsin(X):
    ''' Compute the arcsin of an AutoDiff object and its derivative.

        INPUTS
        ======
        X: an AutoDiff object or constant

        RETURNS
        =======
        A new AutoDiff object or scalar with calculated value and derivative.

        EXAMPLES
        ========
        >>> X = AutoDiff(0.5, 2)
        >>> arcsinAutoDiff = arcsin(X)
        >>> arcsinAutoDiff.val
        0.5235987755982988
        >>> arcsinAutoDiff.der
        2.3094010767585034
    	>>> arcsinAutoDiff.jacobian
        1.1547005383792517
        '''

    try:
        # Is another ADT
        new_val = np.arcsin(X.val) #if (-1 <= X.val and X.val <= 1) else np.nan
        new_der = (1/np.sqrt(1-X.val**2))*X.der #if (-1 < X.val and X.val < 1) else np.na
        new_jacobian = (1/np.sqrt(1-X.val**2))*X.jacobian #if (-1 < X.val and X.val < 1) else np.nan
        
        return AutoDiff(new_val, new_der, X.n, 0, new_jacobian)

    except AttributeError:
		# Constant
        return_val = np.arcsin(X) #if (-1 <= X and X <= 1) else np.nan
        return return_val

# arc cosine
def arccos(X):
    ''' Compute the arccos of an AutoDiff object and its derivative.

        INPUTS
        ======
        X: an AutoDiff object or constant

        RETURNS
        =======
        A new AutoDiff object or scalar with calculated value and derivative.

        EXAMPLES
        ========
        >>> X = AutoDiff(0.5, 2)
        >>> arccosAutoDiff = arccos(X)
        >>> arccosAutoDiff.val
        1.0471975511965976
        >>> arccosAutoDiff.der
        -2.3094010767585034
        >>> arccosAutoDiff.jacobian
        -1.1547005383792517
        '''

    try:
        # Is another ADT
        new_val = np.arccos(X.val) #if (-1 <= X.val and X.val <= 1) else np.nan
        new_der = (-1/np.sqrt(1-X.val**2))*X.der #if (-1 < X.val and X.val < 1) else np.nan
        new_jacobian = (-1/np.sqrt(1-X.val**2))*X.jacobian #if (-1 < X.val and X.val < 1) else np.nan

        return AutoDiff(new_val, new_der, X.n, 0, new_jacobian)

    except AttributeError:
        # Constant
        return_val = np.arccos(X) #if (-1 <= X and X <= 1) else np.nan
        return return_val

# arc tangent
def arctan(X):
    ''' Compute the arctan of an AutoDiff object and its derivative.

        INPUTS
        ======
        X: an AutoDiff object or constant

        RETURNS
        =======
        A new AutoDiff object or scalar with calculated value and derivative.

        EXAMPLES
        ========
        >>> X = AutoDiff(3, 2)
        >>> arctanAutoDiff = arctan(X)
        >>> arctanAutoDiff.val
        1.2490457723982544
        >>> arctanAutoDiff.der
        0.2
        >>> arctanAutoDiff.jacobian
        0.1	
        '''

    try:
        # Is another ADT
        new_val = np.arctan(X.val)
        new_der = (1/(1+X.val**2))*X.der
        new_jacobian = (1/(1+X.val**2))*X.jacobian

        return AutoDiff(new_val, new_der, X.n, 0, new_jacobian)

    except AttributeError:
        # Constant
        return np.arctan(X)


#-------------------HYPERBOLIC TRIG FUNCTIONS-------------------#
# hyperbolic sin
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

# hyperbolic cos
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

# hyperbolic tan
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

#-------------------ARC HYPERBOLIC TRIG FUNCTIONS-------------------#
# hyperbolic arcsin
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
	>>> x = AutoDiff(1.1, 2)
	>>> myAutoDiff = arccosh(x)
	>>> myAutoDiff.val
	0.4435682543851154
	>>> myAutoDiff.der
	(2/np.sqrt(1.1**2 - 1))
	>>> myAutoDiff.jacobian
	(1/np.sqrt(1.1**2 - 1))
	
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
	>>> x = AutoDiff(0.5, 2)
	>>> myAutoDiff = arctanh(x)
	>>> myAutoDiff.val
	0.5493061443340548
	>>> myAutoDiff.der
	2/(1-(0.5)**2)
	>>> myAutoDiff.jacobian
	1/(1-(0.5)**2)
	
	'''
	try:
		new_val = np.arctanh(x.val)
		new_der = ((1)/(1-x.val**2))*x.der
		new_jacobian = ((1)/(1-x.val**2))*x.jacobian
		return AutoDiff(new_val, new_der, x.n, 0, new_jacobian)
	except AttributeError:
		return np.arctanh(x)

#--------------------------EXPONENT FAMILY----------------------------#
#Exponential
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
	>>> x = AutoDiff(10, 2)
	>>> myAutoDiff = exp(x)
	>>> myAutoDiff.val
	22026.465794806718
	>>> myAutoDiff.der
	2*22026.465794806718
	>>> myAutoDiff.jacobian
	22026.465794806718	
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
	x = AutoDiff(4, 2)
	>>> myAutoDiff = log(x)
	>>> myAutoDiff.val
	1.3862943611198906
	>>> myAutoDiff.der
	0.5
	>>> myAutoDiff.jacobian
	0.25
	
	'''
	try:
		new_val = np.log(x.val)
		# Derivative not defined when x = 0
		new_der = (1/(x.val*np.sum(1)))*x.der # if x.val != 0 else None
		new_jacobian = (1/(x.val*np.sum(1)))*x.jacobian # if x.val != 0 else None
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
		new_val = np.log10(x.val)
		# Derivative not defined when x = 0
		new_der = (1/(x.val*np.log(10)))*x.der
		new_jacobian = (1/(x.val*np.log(10)))*x.jacobian
		return AutoDiff(new_val, new_der, x.n, 0, new_jacobian)
	except AttributeError:
		return np.log10(x)

# Square Root
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
		new_val = np.sqrt(x.val)
		new_der = 0.5 * x.val ** (-0.5) * x.der
		new_jacobian = 0.5 * x.val ** (-0.5) * x.jacobian
		return AutoDiff(new_val, new_der, x.n, 0, new_jacobian)
	except AttributeError:
		if x < 0.0:
			warnings.warn('Undefined at value', RuntimeWarning)
			return np.nan
		else:
			return np.sqrt(x)
