# Emma's updated version (fixed overloading)

import numpy as np

class AutoDiff():
	'''
	An auto-differentiation object for scalar functions
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
		self.val = self._convertNonArray(eval_value)
		self.der = self._convertNonArray(der_value)*(self.val**0.0)
		self.n = n
		self.jacobian = self._calcJacobian(k, n, jacobian_value)

	def _convertNonArray(self, value):
		try:
			value.shape
			return value
		except:
			return np.array([[value]])

	def _calcJacobian(self, k, n, jacobian_value):
		if np.all(np.equal(jacobian_value, None)):
			if k != 0:
				rows = self.val.shape[0]
				seed = np.zeros([rows, n])
				seed[:, k-1] = 1
				return seed
		else:
			return jacobian_value*self.val**0.0


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
			return AutoDiff(other.val / self.val, (other.der * self.val - other.val * self.der)/(self.val**2), self.n, 0, (other.jacobian * self.val - other.val * self.jacobian)/(self.val**2) )
		except AttributeError:
			return AutoDiff(other / self.val, other / self.der, self.n, 0, other / self.jacobian)

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
		try:
			# If  AutoDiff of same variable, values and derivatives should both just add
			return AutoDiff(self.val * -1, self.der * -1, self.n, 0, self.jacobian * -1)
		except AttributeError:
			# If constant, just do regular negation
			return self * -1

	def __abs__(self):
		try:
			return AutoDiff(abs(self.val), ((self.val * self.der) / abs(self.val)),
				self.n, 0, ((self.val * self.jacobian) / abs(self.val)))
		except AttributeError:
			return abs(self)
