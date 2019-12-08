import numpy as np
import elemFunctions as ef

class Dual():
	def __init__(self, real, dual):
		self.real = real
		self.dual = dual

	def __str__(self):
		if self.dual >= 0:
			return "{} + {}ε".format(self.real, self.dual)
		else:
			return "{} - {}ε".format(self.real, (self.dual * -1))

	def __add__(self, other):
		try:
			return Dual(self.real + other.real, self.dual + other.dual)
		except AttributeError:
			return Dual(self.real + other, self.dual)

	def __radd__(self, other):
		try:
			return Dual(self.real + other.real, self.dual + other.dual)
		except AttributeError:
			return Dual(self.real + other, self.dual)

	def __sub__(self, other):
		try:
			return Dual(self.real - other.real, self.dual - other.dual)
		except AttributeError:
			return Dual(self.real - other, self.dual)

	def __rsub__(self, other):
		try:
			return Dual(self.real - other.real, self.dual - other.dual)
		except AttributeError:
			return Dual(other - self.real, self.dual)

	def __mul__(self, other):
		try:
			return Dual(self.real * other.real, self.dual * other.real + self.real * other.dual)
		except AttributeError:
			return Dual(self.real * other, self.dual * other)

	def __rmul__(self, other):
		try:
			return Dual(self.real * other.real, self.dual * other.real + self.real * other.dual)
		except AttributeError:
			return Dual(self.real * other, self.dual * other)

	def __truediv__(self, other):
		try:
			return Dual(self.real / other.real, self.dual / other.dual)
		except AttributeError:
			return Dual(self.real / other, (1 / other))

	def __rtruediv__(self, other):
		try:
			return Dual(other.real / self.real, other.dual / self.dual)
		except AttributeError:
			return Dual(other / self.real, self.dual)

	def __pow__(self, other):
		other = float(other) if type(other)==int else other
		try: # need to do
			return Dual(self.real ** other.real, self.dual ** other.dual)
		except AttributeError:
			return Dual(self.real ** other, other * self.dual * (self.real ** (other - 1)))

	def __rpow__(self, other):
		try: # need to do
			return Dual(other.real ** self.real, other.dual ** self.dual)
		except AttributeError:
			return Dual(other ** self.real, self.dual * np.log(other) * (other ** self.real))

	# Unary functions
	def __neg__(self):
		return Dual(-1 * self.real, -1 * self.dual)

	def __abs__(self):
		return self ** (1/2)

	# Comparison
	def __eq__(self, other):
		try:
			if self.dual == other.dual and self.real == other.real:
				return True
			else:
				return False
		except AttributeError:
			return False

	def __ne__(self, other):
		try:
			if self.dual == other.dual and self.real == other.real:
				return False
			else:
				return True
		except AttributeError:
			return True

# x = Dual(3, np.array([1, 0, 0]))
# y = Dual(2, np.array([0, 1, 0]))
x = Dual(3, 1)

f = 2*x**2
print(f)

f_2 = Dual(f.dual, 1)
print(f_2)


