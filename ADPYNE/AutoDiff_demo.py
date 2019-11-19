import numpy as np
from AutoDiff import AutoDiff

print("test 1")
x = AutoDiff(5, 1, 2, 1)
y = AutoDiff(3, 1, 2, 2)

print("x")
print(x.val)
print(x.der)
print(x.jacobian)

print("y")
print(y.val)
print(y.der)
print(y.jacobian)

f = 3*x**2 + 2*y**3
print("f")
print(f.val)
print(f.der)
print(f.jacobian)

print("\ntest 2")
x = AutoDiff(5, 1)
y = AutoDiff(3, 2)

print("x")
print(x.val)
print(x.der)
print(x.jacobian)

print("y")
print(y.val)
print(y.der)
print(y.jacobian)

x = AutoDiff(0, 1)
f = abs(x)
print("f")
print(f.val)
print(f.der)
print(f.jacobian)