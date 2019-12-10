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
			finalString = ""
			episilon = "ε"
			for i, coeff in enumerate(self.coefficients):
				if i == 0:
					finalString += str(coeff)
				elif i == 1:
					theString = " + " + str(coeff) + episilon
					finalString += theString
				else:
					theString = " + " + str(coeff) + episilon + "^" + str(i)
					finalString += theString 
			return finalString

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
		return Dual(-1 * self.Real, -1 * self.Dual)


	# Comparison
	def __eq__(self, other):
		try:
			if self.Dual == other.Dual and self.Real == other.Real:
				return True
			else:
				return False
		except AttributeError:
			return False

	def __ne__(self, other):
		try:
			if self.Dual == other.Dual and self.Real == other.Real:
				return False
			else:
				return True
		except AttributeError:
			return True
