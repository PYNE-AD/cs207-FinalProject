import numpy as np
from AutoDiff import AutoDiff

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
    >>> X = AutoDiff(0.5, 2, 1)
    >>> sinhAutoDiff = sinh(X)
    >>> sinhAutoDiff.val
    0.5210953054937474
    >>> sinhAutoDiff.der
    2.2552519304127614
    >>> sinhAutoDiff.jacobian
    1.1276259652063807

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
    >>> X = AutoDiff(0.5, 2, 1)
    >>> coshAutoDiff = cosh(X)
    >>> coshAutoDiff.val
    1.1276259652063807
    >>> coshAutoDiff.der
    1.0421906109874948
    >>> coshAutoDiff.jacobian
    0.5210953054937474

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
    >>> X = AutoDiff(0.5, 2, 1)
    >>> tanhAutoDiff = tanh(X)
    >>> tanhAutoDiff.val
    0.46211715726000974
    >>> tanhAutoDiff.der
    1.572895465931855
    >>>tanhAutoDiff.jacobian
    0.7864477329659275

    '''
    try:
        val = np.tanh(X.val)
        der = 1/(np.cosh(X.val)**2)*X.der
        jacobian = 1/(np.cosh(X.val)**2)*X.jacobian
        return AutoDiff(val, der, X.n, 0, jacobian)
    except AttributeError:
        return np.tanh(X)

# log base 10
def log10(X):
    ''' Compute the natural log of an AutoDiff object and its derivative.

    INPUTS
    ======
    X: an AutoDiff object

    RETURNS
    =======
    A new AutoDiff object with calculated value and derivative.

    EXAMPLES
    ========
    >>> X = AutoDiff(0.5, 2, 1)
    >>> myAutoDiff = log(X)
    >>> myAutoDiff.val
    -0.3010299956639812
    >>> myAutoDiff.der
    1.737177927613007
    >>>myAutoDiff.jacobian
    0.8685889638065035

    '''
    try:
        val = np.log10(X.val)
        # Derivative not defined when X = 0
        der = (1/(X.val*np.log(10)))*X.der if X.val != 0 else None
        jacobian = (1/(X.val*np.log(10)))*X.jacobian if X.val != 0 else None
        return AutoDiff(val, der, X.n, 0, jacobian)
    except AttributeError:
        return np.log10(X)
