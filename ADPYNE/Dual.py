import numpy as np
import ADPYNE.elemFunctions as ef

class Dual():
	def __init__(self, Real, Dual):
		self.Real = Real
		self.Dual = Dual

	def __str__(self):
		if self.Dual >= 0:
			return "{} + {}ε".format(self.Real, self.Dual)
		else:
			return "{} - {}ε".format(self.Real, (self.Dual * -1))

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
			return Dual(other - self.Real, self.Dual)

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
			return Dual(self.Real / other.Real, self.Dual / other.Dual)
		except AttributeError:
			return Dual(self.Real / other, (1 / other))

	def __rtruediv__(self, other):
		try:
			return Dual(other.Real / self.Real, other.Dual / self.Dual)
		except AttributeError:
			return Dual(other / self.Real, self.Dual)

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
		return Dual(-1 * self.Real, -1 * self.Dual)

	def __abs__(self):
		return self ** (1/2)

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

# x = Dual(3, np.array([1, 0, 0]))
# y = Dual(2, np.array([0, 1, 0]))
# x = Dual(3, 1)

# f = 2*x**2
# print(f)

# f_2 = Dual(f.Dual, 1)
# print(f_2)


