import numpy as np

#HYPERBOLIC TRIG FUNCTIONS
def sinh(X):
        ''' Compute the sinh of an AutoDiff object and its derivative.

        INPUTS
        ======
        X: an AutoDiff object

        RETURNS
        =======
        A new AutoDiff object with calculated value and derivative.

        EXAMPLES
        ========
        >>> sinhAutoDiff = sinh(X)
        >>> sinhAutoDiff.val
        np.sinh(X.val)
        >>> sinhAutoDiff.der
        np.cosh(X.val)*X.der
        >>> sinhAutoDiff.jacobian
        np.cosh(X.val)*X.jacobian

        '''
    try:
        val = np.sinh(X.val)
        der = np.cosh(X.val)*X.der
        jacobian = np.cosh(X.val)*X.jacobian
        return AutoDiff(val, der, X.n, 0, jacobian)
    except AttributeError:
        return np.sinh(X)

def cosh(X):
        ''' Compute the cosh of an AutoDiff object and its derivative.

        INPUTS
        ======
        X: an AutoDiff object

        RETURNS
        =======
        A new AutoDiff object with calculated value and derivative.

        EXAMPLES
        ========
        >>> coshAutoDiff = cosh(X)
        >>> coshAutoDiff.val
        np.cosh(X.val)
        >>> coshAutoDiff.der
        np.sinh(X.val)*X.der
        >>> coshAutoDiff.jacobian
        p.cosh(X.val)*X.jacobian
        

        '''
    try:
        val = np.cosh(X.val)
        der = np.sinh(X.val)*X.der
        jacobian = np.sinh(X.val)*X.jacobian
        return AutoDiff(val, der, X.n, 0, jacobian)
    except AttributeError:
        return np.cosh(X)

def tanh(X):
        ''' Compute the tanh of an AutoDiff object and its derivative.

        INPUTS
        ======
        X: an AutoDiff object

        RETURNS
        =======
        A new AutoDiff object with calculated value and derivative.

        EXAMPLES
        ========
        >>> tanhAutoDiff = tanh(X)
        >>> tanhAutoDiff.val
        np.tanh(X.val)
        >>> tanhAutoDiff.der
        1/(np.cosh(X.val)**2)*X.der
        >>>tanhAutoDiff.jacobian
        1/(np.cosh(X.val)**2)*X.jacobian

        '''
    try:
        val = np.tanh(X.val)
        der = 1/(np.cosh(X.val)**2)*X.der
        jacobian = 1/(np.cosh(X.val)**2)*X.jacobian
        return AutoDiff(val, der, X.n, 0, jacobian)
    except AttributeError:
        return np.tanh(X)
