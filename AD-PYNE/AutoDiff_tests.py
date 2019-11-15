import numpy as np
from AutoDiff import AutoDiff

#x = AutoDiff(np.array([[5, 1, 3]]).T, np.array([[1, 1]]), 2, 1)
x = AutoDiff(5, 1)
#y = AutoDiff(np.array([[2, -1, 3]]).T, np.array([[1, 1]]), 2, 2)
y = AutoDiff(3, 1)
# z = AutoDiff(np.array([[1, 0]]).T, np.array([[1, 1]]), 3, 3)

print("x")
print(x.val)
print(x.der)
print(x.jacobian)

print("y")
print(y.val)
print(y.der)
print(y.jacobian)

# f = 3*x**2 + 2*y**3
# print("f")
# print(f.val)
# print(f.der)
# print(f.jacobian)