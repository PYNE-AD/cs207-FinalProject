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
        >>> X = AutoDiff(0.5, 2)
        >>> arcsinAutoDiff = arcsin(X)
        >>> arcsinAutoDiff.val
        0.5235987755982988
        >>> arcsinAutoDiff.der
        2.3094010767585034
    	>>> arcsinAutoDiff.jacobian
        1.1547005383792517
        '''

    try:
        # Is another ADT
        new_val = np.arcsin(X.val) #if (-1 <= X.val and X.val <= 1) else np.nan
        new_der = (1/np.sqrt(1-X.val**2))*X.der #if (-1 < X.val and X.val < 1) else np.na
        new_jacobian = (1/np.sqrt(1-X.val**2))*X.jacobian #if (-1 < X.val and X.val < 1) else np.nan
        
        return AutoDiff(new_val, new_der, X.n, 0, new_jacobian)

    except AttributeError:
		# Constant
        return_val = np.arcsin(X) #if (-1 <= X and X <= 1) else np.nan
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
        >>> X = AutoDiff(0.5, 2)
        >>> arccosAutoDiff = arccos(X)
        >>> arccosAutoDiff.val
        1.0471975511965976
        >>> arccosAutoDiff.der
        -2.3094010767585034
        >>> arccosAutoDiff.jacobian
        -1.1547005383792517
        '''

    try:
        # Is another ADT
        new_val = np.arccos(X.val) #if (-1 <= X.val and X.val <= 1) else np.nan
        new_der = (-1/np.sqrt(1-X.val**2))*X.der #if (-1 < X.val and X.val < 1) else np.nan
        new_jacobian = (-1/np.sqrt(1-X.val**2))*X.jacobian #if (-1 < X.val and X.val < 1) else np.nan

        return AutoDiff(new_val, new_der, X.n, 0, new_jacobian)

    except AttributeError:
        # Constant
        return_val = np.arccos(X) #if (-1 <= X and X <= 1) else np.nan
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
        >>> X = AutoDiff(3, 2)
        >>> arctanAutoDiff = arctan(X)
        >>> arctanAutoDiff.val
        1.2490457723982544
        >>> arctanAutoDiff.der
        0.2
        >>> arctanAutoDiff.jacobian
        0.1	
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
