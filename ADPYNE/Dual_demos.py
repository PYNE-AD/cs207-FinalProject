import numpy as np
from Dual import Dual
import elemFunctions as ef

# Negation
# print("Reverse subtraction")
# x = Dual(2, 1)
# f = 1 - x
# print(f)

# x2 = x.makeHighestOrder(4)
# f = 1 - x2
# f.buildCoefficients(4)
# print(f)

# Negative power
# print("Negative power")
# x = Dual(2, 1)
# f = x**-5
# print(f)

# x2 = x.makeHighestOrder(2)
# f = x2**-5
# f.buildCoefficients(2)
# print(f)

# Reverse divison
# print("\nReverse division")
# x = Dual(3, np.array([1, 0]))
# y = Dual(6, np.array([0, 1]))

# f = y/x
# print(f)

# Elementary Functions

# x = Dual(1/np.sqrt(2), 1)
# f = (1/ef.sqrt(1-(x**2)))
# print(f)


# x2 = x.makeHighestOrder(2)
# f = (1/ef.sqrt(1-(x2**2)))
# f.buildCoefficients(2)
# print(f)

# f.buildCoefficients(2)
# print(f)

# Arc Sin
# print("\nArc Sine")
# x = Dual(0.5, 1)
# f = ef.arcsin(x)
# print(f)

# x5 = x.makeHighestOrder(3)
# f = ef.arcsin(x5)
# f.buildCoefficients(3)
# print(f)

# Arc Cosine
# print("\nArc Cosine")
# x = Dual(0.5, 1)
# f = ef.arccos(x)
# print(f)

# x5 = x.makeHighestOrder(3)
# f = ef.arccos(x5)
# f.buildCoefficients(3)
# print(f)

# Arc Tangent
# print("\nArc Tangent")
# x = Dual(0.5, 1)
# f = ef.arctan(x)
# print(f)

# x5 = x.makeHighestOrder(3)
# f = ef.arctan(x5)
# f.buildCoefficients(3)
# print(f)

# Hyperbolic Sine
# print("\nHyperbolic Sine")
# x = Dual(0.5, 1)
# f = ef.sinh(x)
# print(f)

# x5 = x.makeHighestOrder(3)
# f = ef.sinh(x5)
# f.buildCoefficients(3)
# print(f)

# # Hyberpolic Cosine
# print("\nHyperbolic Cosine")
# x = Dual(0.5, 1)
# f = ef.cosh(x)
# print(f)

# x5 = x.makeHighestOrder(3)
# f = ef.cosh(x5)
# f.buildCoefficients(3)
# print(f)

# Hyberpolic Tangent
# print("\nHyperbolic Tangent")
# x = Dual(0.5, 1)
# f = ef.tanh(x)
# print(f)

# x5 = x.makeHighestOrder(3)
# f = ef.tanh(x5)
# f.buildCoefficients(3)
# print(f)

# Hyberpolic Arcsine
# print("\nHyberpolic Arcsine")
# x = Dual(0.5, 1)
# f = ef.arcsinh(x)
# print(f)

# x5 = x.makeHighestOrder(3)
# f = ef.arcsinh(x5)
# f.buildCoefficients(3)
# print(f)

# Hyberpolic Cosine
# print("\nHyberpolic Cosine")
# x = Dual(2, 1)
# f = ef.arccosh(x)
# print(f)

# x5 = x.makeHighestOrder(3)
# f = ef.arccosh(x5)
# f.buildCoefficients(3)
# print(f)

# # Hyberpolic Tangent
# print("\nHyberpolic Tangent")
# x = Dual(0.5, 1)
# f = ef.arctanh(x)
# print(f)

# x5 = x.makeHighestOrder(3)
# f = ef.arctanh(x5)
# f.buildCoefficients(3)
# print(f)

# Natural exponential
# print("\nNatural exponential")
# x = Dual(2, 1)
# f = ef.exp(3*x)
# print(f)

# x5 = x.makeHighestOrder(3)
# f = ef.exp(3*x5)
# f.buildCoefficients(3)
# print(f)

# Natural log
# print("\nNatural log")
# x = Dual(2, 1)
# f = ef.log(x)
# print(f)

# x5 = x.makeHighestOrder(3)
# f = ef.log(x5)
# f.buildCoefficients(3)
# print(f)

# # Log any base
# print("\nLog any base")
# x = Dual(2, 1)
# f = ef.logbase(x, 7)
# print(f)

# x5 = x.makeHighestOrder(3)
# f = ef.logbase(x5, 7)
# f.buildCoefficients(3)
# print(f)

# Log base 10
# print("\nLog base 10")
# x = Dual(2, 1)
# f = ef.log10(x)
# print(f)

# x5 = x.makeHighestOrder(3)
# f = ef.log10(x5)
# f.buildCoefficients(3)
# print(f)

# print(f.Real.Real)
# print(f.Real.Dual)
# print(f.Dual.Real)
# print(f.Dual.Dual)

# print("mess")
# print(f)
# f.buildCoefficients(2)
# print(f)

# # Arc Cosine
# print("\nArc Cosine")
# x = Dual(0.5, 1)
# f = ef.arccos(x)
# print(f)

# x5 = x.makeHighestOrder(4)
# f = ef.arccos(x5)
# f.buildCoefficients(4)
# print(f)

# # Arc Tangent
# print("\nArc Tangent")
# x = Dual(0.5, 1)
# f = ef.arctan(x)
# print(f)

# x5 = x.makeHighestOrder(4)
# f = ef.arctan(x5)
# f.buildCoefficients(4)
# print(f)

# Hypberbolic sin
# print("Hyberbolic sin")
# x = Dual(0.5, 1)
# f = ef.arctan(x)
# print(f)

# x5 = x.makeHighestOrder(5)
# f = ef.arctan(x5)
# f.buildCoefficients(5)
# print(f)

# Square root
# print("Square root")
# x = Dual(4, 1)
# f = ef.sqrt(x)
# print(f)

# x5 = x.makeHighestOrder(4)
# f = ef.sqrt(x5)
# f.buildCoefficients(4)
# print(f)

# f = ef.tan(x_3)
# print(f)

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

# x = Dual(np.pi/4, 1)
# f = ef.tan(x)

# # # print(x)
# # # f = 3*x**3

# x_3 = x.makeHighestOrder(5)
# f = ef.tan(x_3)
# print(f)

# f.buildCoefficients(5)
# print(f)

# f = x_3**2

# # print(f.Real.Real.Real)
# f.buildCoefficients(10)
# print(f.coefficients)

