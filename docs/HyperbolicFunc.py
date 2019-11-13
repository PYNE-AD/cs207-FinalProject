import numpy as np

#HYPERBOLIC TRIG FUNCTIONS
def sinh(X):
        ''' Compute the sinh of an AutoDiff object and its derivative.

        INPUTS
        ======
        x: an AutoDiff object

        RETURNS
        =======
        A new AutoDiff object with calculated value and derivative.

        EXAMPLES
        ========
        >>> sinhAutoDiff = sinh(x)
        >>> sinhAutoDiff.val
        np.sinh(x.val)
        >>> sinhAutoDiff.der
        np.cosh(x.val)*x.der

        '''
    try:
        val = np.sinh(X.val)
        der = np.cosh(X.val)*X.der
        return AutoDiff(val, der)
    except AttributeError:
        return np.sinh(X)

def cosh(X):
        ''' Compute the cosh of an AutoDiff object and its derivative.

        INPUTS
        ======
        x: an AutoDiff object

        RETURNS
        =======
        A new AutoDiff object with calculated value and derivative.

        EXAMPLES
        ========
        >>> coshAutoDiff = cosh(x)
        >>> coshAutoDiff.val
        np.cosh(x.val)
        >>> coshAutoDiff.der
        np.sinh(x.val)*x.der

        '''
    try:
        val = np.cosh(X.val)
        der = np.sinh(X.val)*X.der
        return AutoDiff(val, der)
    except AttributeError:
        return np.cosh(X)

def tanh(X):
        ''' Compute the tanh of an AutoDiff object and its derivative.

        INPUTS
        ======
        x: an AutoDiff object

        RETURNS
        =======
        A new AutoDiff object with calculated value and derivative.

        EXAMPLES
        ========
        >>> tanhAutoDiff = tanh(x)
        >>> tanhAutoDiff.val
        np.tanh(x.val)
        >>> coshAutoDiff.der
        1/(np.cosh(X.val)**2)*X.der

        '''
    try:
        val = np.tanh(X.val)
        der = 1/(np.cosh(X.val)**2)*X.der
        return AutoDiff(val, der)
    except AttributeError:
        return np.tanh(X)
