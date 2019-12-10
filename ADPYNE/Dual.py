import numpy as np

class Dual():
	def __init__(self, Real, Dual = 1):
		self.Real = Real
		self.Dual = Dual
		self.coefficients = []

	def makeHighestOrder(self, order):
		theLongdual = self._createNestedduals(self.Real, self.Dual, order)**1
		return theLongdual

	def _createNestedduals(self, value, dual, order):
		if order == 1:
			return Dual(value)
		else:
			return Dual(self._createNestedduals(value, dual, order - 1), dual)

	def buildCoefficients(self, n):
		coeffs = []
		for i in range(n + 1):
			theCoeff = self
			# get duals
			for j in range(i):
				theCoeff = theCoeff.Dual
			# get Reals
			for k in range(n-i):
				theCoeff = theCoeff.Real
			coeffs.append(theCoeff)
		self.coefficients = coeffs

	def __str__(self):
		if len(self.coefficients) == 0:
			return "{} + {}ε".format(self.Real, self.Dual)
		else:
			return "{}".format(self.coefficients)

	def __add__(self, other):
		try:
			return Dual(self.Real + other.Real, self.Dual + other.Dual)
		except AttributeError:
			return Dual(self.Real + other, self.Dual)

	def __radd__(self, other):
		try:
			return Dual(self.Real + other.Real, self.Dual + other.Dual)
		except AttributeError:
			return Dual(self.Real + other, self.Dual)

	def __sub__(self, other):
		try:
			return Dual(self.Real - other.Real, self.Dual - other.Dual)
		except AttributeError:
			return Dual(self.Real - other, self.Dual)

	def __rsub__(self, other):
		try:
			return Dual(self.Real - other.Real, self.Dual - other.Dual)
		except AttributeError:
			return Dual(other - self.Real, -1 * self.Dual)


	def __mul__(self, other):
		try:
			return Dual(self.Real * other.Real, self.Dual * other.Real + self.Real * other.Dual)
		except AttributeError:
			return Dual(self.Real * other, self.Dual * other)

	def __rmul__(self, other):
		try:
			return Dual(self.Real * other.Real, self.Dual * other.Real + self.Real * other.Dual)
		except AttributeError:
			return Dual(self.Real * other, self.Dual * other)

	def __truediv__(self, other):
		try:
			return Dual(self.Real / other.Real, (self.Dual*other.Real - self.Real*other.Dual) / other.Real**2)
		except AttributeError:
			return Dual(self.Real / other, (self.Dual / other))

	def __rtruediv__(self, other):
		try:
			return Dual(other.Real / self.Real, (other.Dual*self.Real - other.Real*self.Dual) / self.Real**2)
		except AttributeError:
			return Dual(other / self.Real, -1.0*((other*self.Dual)/(self.Real**2)))

	def __pow__(self, other):
		other = float(other) if type(other)==int else other
		try: # need to do
			return Dual(self.Real ** other.Real, self.Dual ** other.Dual)
		except AttributeError:
			return Dual(self.Real ** other, other * self.Dual * (self.Real ** (other - 1)))

	def __rpow__(self, other):
		try: # need to do
			return Dual(other.Real ** self.Real, other.Dual ** self.Dual)
		except AttributeError:
			return Dual(other ** self.Real, self.Dual * np.log(other) * (other ** self.Real))

	# Unary functions
	def __neg__(self):
		try:
			return Dual(-1.0 * self.Real, -1.0 * self.Dual)
		except:
			return -1.0 * self

	def __abs__(self):
		return Dual(abs(self.Real), self.Dual/abs(self.Real))

	def __pos__(self):
		try:
			return Dual(self.Real, self.Dual)
		except:
			return self

	# Comparison
	def __eq__(self, other):
		try:
			if np.all(self.Dual == other.Dual) and np.all(self.Real == other.Real):
				return True
			else:
				return False
		except AttributeError:
			return False

	def __ne__(self, other):
		try:
			if np.all(self.Dual == other.Dual) and np.all(self.Real == other.Real):
				return False
			else:
				return True
		except AttributeError:
			return True

def vectorizeDual(functions, order):
	n = len(functions)
	m = order + 1
	vector = np.zeros([n, m])
	for i, f in enumerate(functions):
   	 f.buildCoefficients(order)
   	 for j, coeff in enumerate(f.coefficients):
   		 vector[i,j] = coeff

	# Create a copy of one function
	f_everything = functions[0]
	f_everything.coefficients = np.zeros([n, m])
	for i in range(m):
   	 f_everything.coefficients[:, i] = vector[:, i]

	return f_everything


def makeHessianVars(x,y):
	xh = x.makeHighestOrder(3)
	yh = y.makeHighestOrder(3)
	return xh, yh