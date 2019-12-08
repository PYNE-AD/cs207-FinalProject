import numpy as np

class AutoDiff():
	'''
	An auto-differentiation object for scalar and vector functions
	'''

	def __init__(self, eval_value, der_value, n=1, k=1, jacobian_value = np.array([[None]])):
		'''
		INPUTS
		======
		eval_value: 	value to be evaluated at
		der_value: 		evaluated value of the derivative
		n:				number of input variables the final function will use
		k:				denotes that the autodiff uses the kth input variable in a vector of 1 to n variables
		jacob_value: 	denotes the evaluted value of the jacobian

		RETURNS
		=======
		An AutoDiff object with calculated value, derivative, and jacobian.

		EXAMPLES
		========
		>>> myAutoDiff = AutoDiff(3, 2)
		>>> myAutoDiff.val
		3
		>>> myAutoDiff.der
		2
		>>> myAutoDiff.jacobian
		1
		'''
		# Convert int or float to array
		self.val = self._convertNonArray(eval_value, k)
		self.jacobian = self._calcJacobian(k, n, jacobian_value)
		self.der = self._calcDerivative(der_value, k)
		self.n = n

	def _convertNonArray(self, value, k):
		# try:
		# 	value.shape
		# 	return value
		# except:
		# 	return np.array([[value]])
		if k != 0:
			try:
				return np.array(value).reshape(len(value), 1)
			except:
				return np.array([[value]])
		else:
			return value

	def _calcJacobian(self, k, n, jacobian_value):
		if np.all(np.equal(jacobian_value, None)):
			if k != 0:
				rows = self.val.shape[0]
				seed = np.zeros([rows, n])
				seed[:, k-1] = 1
				return seed
		else:
			return jacobian_value

	def _calcDerivative(self, der_value, k):
		if k != 0:
			jacobian = (np.array([self.jacobian[0]]))
			der = self._convertNonArray(der_value, k)
			return np.dot(der, jacobian)
		else:
			return self._convertNonArray(der_value, k)

	def __str__(self):
		return "Value: {}, Derivative: {}, Jacobian: {}".format(self.val, self.der, self.jacobian)

	def __repr__(self):
		return "Value: {}, Derivative: {}, Jacobian: {}".format(self.val, self.der, self.jacobian)

	def __add__(self, other):
		try:
			# If  AutoDiff of same variable, values and derivatives should both just add
			return AutoDiff(self.val + other.val, self.der + other.der, self.n, 0, self.jacobian + other.jacobian)
		except AttributeError:
			# If trying to add a constant to AutoDiff, only add values. Constant has derivative of 0 so no addition needed.
			return AutoDiff(self.val + other, self.der, self.n, 0, self.jacobian)

	# Account for reverse addition
	def __radd__(self, other):
		try:
			return AutoDiff(self.val + other.val, self.der + other.der, self.n, 0, self.jacobian + other.jacobian)
		except AttributeError:
			return AutoDiff(self.val + other, self.der, self.n, 0, self.jacobian)

	def __sub__(self, other):
		try:
			# If  AutoDiff of same variable, values and derivatives should both just add
			return AutoDiff(self.val - other.val, self.der - other.der, self.n, 0, self.jacobian - other.jacobian)
		except AttributeError:
			# If trying to add a constant to AutoDiff, only add values. Constant has derivative of 0 so no subtraction needed.
			return AutoDiff(self.val - other, self.der, self.n, 0, self.jacobian)

	# Account for reverse subtraction
	def __rsub__(self, other):
		try:
			return AutoDiff(other.val - self.val, other.der - self.der, self.n, 0, other.jacobian - self.jacobian)
		except AttributeError:
			return AutoDiff(other - self.val, self.der, self.n, 0, self.jacobian)

	def __mul__(self, other):
		try:
			# Use product rule
			return AutoDiff(self.val * other.val, self.val * other.der + self.der * other.val, self.n, 0, self.val * other.jacobian + self.jacobian * other.val)
		except AttributeError:
			return AutoDiff(self.val * other, self.der * other, self.n, 0, self.jacobian * other)

	# Account for reverse multiplication
	def __rmul__(self, other):
		try:
			# Use product rule
			return AutoDiff(self.val * other.val, self.val * other.der + self.der * other.val, self.n, 0, self.val * other.jacobian + self.jacobian * other.val)
		except AttributeError:
			return AutoDiff(self.val * other, self.der * other, self.n, 0, self.jacobian * other)

	def __truediv__(self, other):
		try:
			# Use quotient rule
			return AutoDiff(self.val / other.val, (self.der * other.val - self.val * other.der)/(other.val**2), self.n, 0, (self.jacobian * other.val - self.val * other.jacobian)/(other.val**2))
		except AttributeError:
			return AutoDiff(self.val / other, self.der / other, self.n, 0, self.jacobian / other)

	# Account for reverse true division
	def __rtruediv__(self, other):
		try:
			# Use quotient rule
			# other/self
			return AutoDiff(other.val / self.val, (self.val * other.der - other.val * self.der)/(self.val**2), self.n, 0, (other.jacobian * self.val - other.val * self.jacobian)/(self.val**2) )
		except AttributeError:
			print('here')
			return AutoDiff(other / self.val, (self.val * 0 - other * self.der)/(self.val**2), self.n, 0, (self.val * 0 - other * self.jacobian)/(self.val**2))

	def __pow__(self, other):
		# Convert to float so that negative integers will work
		other = float(other) if type(other)==int else other
		try:
			return AutoDiff(self.val**other.val, other.val * (self.val ** (other.val-1)) * self.der + (self.val**other.val) *np.log(np.abs(self.val)) * other.der, self.n, 0, other.val * (self.val**(other.val-1)) * self.jacobian + (self.val**other.val * np.log(np.abs(self.val)) * other.jacobian))
		except AttributeError:
			return AutoDiff(self.val**other, other * (self.der) * self.val**(other-1), self.n, 0, other * (self.jacobian) * self.val**(other-1))

	def __rpow__(self, other):
		try:
			return AutoDiff(other.val**self.val, other.val * (self.val ** (other.val-1)) * self.der + (self.val**other.val) *np.log(np.abs(self.val)) * other.der, self.n, 0, other.val * (self.val**(other.val-1)) * self.jacobian + (self.val**other.val * np.log(np.abs(self.val)) * other.jacobian))
		except AttributeError:
			return AutoDiff(other**self.val, np.log(other) * other**self.val * self.der, self.n, 0, np.log(other) * other**self.val * self.jacobian)

	# Unary operations
	# Unary addition: identity
	def __pos__(self):
		return self

	# Unary subtration: negation
	def __neg__(self):
		# If  AutoDiff of same variable, values and derivatives should both just add
		return AutoDiff(self.val * -1, self.der * -1, self.n, 0, self.jacobian * -1)

	def __abs__(self):
		return AutoDiff(abs(self.val), ((self.val * self.der) / abs(self.val)),
				self.n, 0, ((self.val * self.jacobian) / abs(self.val)))

	def __invert__(self):
		return AutoDiff(~self.val, self.der * -1, self.n, 0, self.jacobian * -1)

	def __eq__(self, other):
		try:
			if self.val == other.val and self.der == other.der:
				return True
			else:
				return False
		except AttributeError:
			return False

	def __ne__(self, other):
		try:
			if self.val == other.val and self.der == other.der:
				return False
			else:
				return True
		except AttributeError:
			return True

def vectorize(ad_functions, n_vector, n_inputs):
	'''
	INPUTS
	======
	ad_functions: 	a list or row vector of AutoDiff objects
	n_inputs:		number of input variables the final function will use

	RETURNS
	=======
	An AutoDiff object of vector functions with calculated value, derivative, and jacobian.

	EXAMPLES
	========
    >>> x = AutoDiff(3, np.array([[2, 0, 0]]), n=3, k=1)
    >>> y = AutoDiff(2, np.array([[0, 2, 0]]), n=3, k=2)
    >>> z = AutoDiff(-1, np.array([[0, 0, 2]]), n=3, k=3)
    >>> fs = [3*x + 2*y + 4*z, x - y + z, x/2, 2*x - 2*y]
    >>> f = vectorize(fs, 3)
	>>> f.val
	np.array([[9],[0],[1.5],[2]]
	>>> f.der
	np.array([[6, 4, 8], [2, -2, 2], [1, 0, 0], [4, -4, 0]])
	>>> f.jacobian
	np.array([[3, 2, 4], [1, -1, 1], [0.5, 0, 0], [2, -2, 0]])
	'''
	if n_vector == 1:
		val = np.zeros([len(ad_functions), n_vector])
		der = np.zeros([len(ad_functions), n_inputs])
		jacobian = np.zeros([len(ad_functions), n_inputs])
		for i, f in enumerate(ad_functions):
			val[i, :] = f.val
			der[i, :] = f.der
			jacobian[i, :] = f.jacobian
	else:
		val = np.zeros([len(ad_functions), n_vector])
		der = np.zeros([n_vector, len(ad_functions), n_inputs])
		jacobian = np.zeros([n_vector, len(ad_functions), n_inputs])
		for j in range(n_vector):
			der_j = der[j]
			jac_j = jacobian[j]
			for i, f in enumerate(ad_functions):
				val[i, :] = f.val.T
				der_j[i, :] = f.der[j]
				jac_j[i, :]	= f.jacobian[j]

	return AutoDiff(val, der, n_inputs, 0, jacobian)
