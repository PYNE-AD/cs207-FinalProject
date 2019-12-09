import numpy as np
from Dual import Dual
import elemFunctions as ef

# x = Dual(3, np.array([1, 0]))
# y = Dual(5, np.array([0, 1]))

# f = x*y

# print(f)

# x_3 = x.makeHighestOrder(3)
# y_3 = y.makeHighestOrder(3)

# f = x_3*y_3
# print(f)
# f.buildCoefficients(3)
# print(f.coefficients)

# x = Dual(1, np.array([1, 0]))
# y = Dual(2, np.array([0, 1]))

# f = x/y

# print(f)

# x = Dual(2, 2)
# f = x/5
# print(f)

# x = Dual(Dual(Dual(3, 1), 1), 1)
# f = 3*x**2
# print(f)
# print(f.Real)
# print(f.Dual)
# print(f.Real.Real)
# print(f.Real.Dual)
# print(f.Dual.Real)
# print(f.Dual.Dual)

# print("")

x = Dual(np.pi/4, 1)
f = ef.tan(x)

# # print(x)
# # f = 3*x**3

x_3 = x.makeHighestOrder(5)
f = ef.tan(x_3)
print(f)

f.buildCoefficients(5)
print(f)

# f = x_3**2

# # print(f.Real.Real.Real)
# f.buildCoefficients(10)
# print(f.coefficients)

# f._getReal(3)

# print(f.Real.Real.Dual)
# print(f.Real.Dual.Real)
# print(f.Dual.Real.Real)
# print(f.Real.Dual.Dual)
# print(f.Dual.Dual.Real)
# print(f.Dual.Real.Dual)
# print(f.Dual.Dual.Dual)

# x_5 = x.makeHighestOrder(5)
# f = 3*x_5**2
# # print(type(f))
# # print(f.Real)
# # print(f.Dual)
# # print(f.Real.Real)
# # print(f.Real.Dual)
# # print(f.Dual.Real)
# # print(f.Dual.Dual)
# print(f.Real.Real.Real.Real.Real)
