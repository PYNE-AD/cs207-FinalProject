# Emma's Elementary Functions (sin, cos, tan)

import numpy as np
import warnings
from AutoDiff import AutoDiff

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


# Paulina's square root function, Emma's edits
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
