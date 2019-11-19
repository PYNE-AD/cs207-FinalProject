import numpy as np
from AutoDiff import AutoDiff


def arcsin(X):
    ''' Compute the arcsin of an AutoDiff object and its derivative.

        INPUTS
        ======
        X: an AutoDiff object or constant

        RETURNS
        =======
        A new AutoDiff object or scalar with calculated value and derivative.

        EXAMPLES
        ========
        >>> arcsinAutoDiff = arcsin(X)
        >>> arcsinAutoDiff.val
        np.arcsin(X.val)
        >>> arcsinAutoDiff.der
        1/np.sqrt(1-X.val**2)*X.der
    	>>> arcsinAutoDiff.jacobian
        1/np.sqrt(1-X.val**2)*X.jacobian
        '''

    try:
        # Is another ADT
        new_val = np.arcsin(X.val) if (-1 <= X.val and X.val <= 1) else np.nan
        new_der = (1/np.sqrt(1-X.val**2))*X.der if (-1 < X.val and X.val < 1) else np.nan
        new_jacobian = (1/np.sqrt(1-X.val**2))*X.jacobian if (-1 < X.val and X.val < 1) else np.nan
        new_ans = AutoDiff(new_val, new_der, X.n, 0, new_jacobian)
        
        return new_ans

    except AttributeError:
		# Constant
        return_val = np.arcsin(X) if (-1 <= X and X <= 1) else np.nan
        return return_val


def arccos(X):
    ''' Compute the arccos of an AutoDiff object and its derivative.

        INPUTS
        ======
        X: an AutoDiff object or constant

        RETURNS
        =======
        A new AutoDiff object or scalar with calculated value and derivative.

        EXAMPLES
        ========
        >>> arccosAutoDiff = arccos(X)
        >>> arccosAutoDiff.val
        np.arccos(X.val)
        >>> arccosAutoDiff.der
        -1/np.sqrt(1-X.val**2)*X.der
    >>> arccosAutoDiff.jacobian
        -1/np.sqrt(1-X.val**2)*X.jacobian
        '''

    try:
        # Is another ADT
        new_val = np.arccos(X.val) if (-1 <= X.val and X.val <= 1) else np.nan
        new_der = (-1/np.sqrt(1-X.val**2))*X.der if (-1 < X.val and X.val < 1) else np.nan
        new_jacobian = (-1/np.sqrt(1-X.val**2))*X.jacobian if (-1 < X.val and X.val < 1) else np.nan

        return AutoDiff(new_val, new_der, X.n, 0, new_jacobian)

    except AttributeError:
        # Constant
        return_val = np.arccos(X) if (-1 <= X and X <= 1) else np.nan
        return return_val


def arctan(X):
    ''' Compute the arctan of an AutoDiff object and its derivative.

        INPUTS
        ======
        X: an AutoDiff object or constant

        RETURNS
        =======
        A new AutoDiff object or scalar with calculated value and derivative.

        EXAMPLES
        ========
        >>> arctanAutoDiff = arctan(X)
        >>> arctanAutoDiff.val
        np.arctan(X.val)
        >>> arctanAutoDiff.der
        1/np.sqrt(1+X.val**2)*X.der
        >>> arctanAutoDiff.jacobian
        1/np.sqrt(1+X.val**2)*X.jacobian	
        '''

    try:
        # Is another ADT
        new_val = np.arctan(X.val)
        new_der = (1/(1+X.val**2))*X.der
        new_jacobian = (1/(1+X.val**2))*X.jacobian

        return AutoDiff(new_val, new_der, X.n, 0, new_jacobian)

    except AttributeError:
        # Constant
        return np.arctan(X)
