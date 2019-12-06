import numpy as np

# Dual class to get the dual number
class Dual():
    def __init__(self, order_of_derivative, x_value):
        self.order_of_derivative = order_of_derivative
        self.x_value = x_value
        
    def get_matrix(self):
        n = self.order_of_derivative
        A = self.x_value*np.identity(n)
        A[0, n-1] = 1
        return A