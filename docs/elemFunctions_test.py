import pytest
import elemFunctions

def test_sinh_results():
    sinhAutoDiff = sinh(X)    
    assert sinhAutoDiff.val == np.sinh(X.val)
    assert sinhAutoDiff.der == np.cosh(X.val)*X.der
    assert sinhAutoDiff.jacobian == np.cosh(X.val)*X.jacobian
    
def test_sinh_types():
    with pytest.raises(TypeError):
        sinh(1)
               
def test_cosh_results():
    coshAutoDiff = sinh(X)    
    assert coshAutoDiff.val == np.cosh(X.val)
    assert coshAutoDiff.der == np.sinh(X.val)*X.der
    assert coshAutoDiff.jacobian == np.sinh(X.val)*X.jacobian
    
def test_cosh_types():
    with pytest.raises(TypeError):
        cosh(1)  
        
def test_tanh_results():
    tanhAutoDiff = sinh(X)    
    assert tanhAutoDiff.val == np.tanh(X.val)
    assert tanhAutoDiff.der == 1/(np.cosh(X.val)**2)*X.der
    assert tanhAutoDiff.jacobian == 1/(np.cosh(X.val)**2)*X.jacobian
    
def test_tanh_types():
    with pytest.raises(TypeError):
        tanh(1)
