import warnings
import pytest
import numpy as np
from AutoDiff import AutoDiff
import elemFunctions as ef
    
# hyperbolic sine tests
def test_sinh_results():
    X = AutoDiff(0.5, 2)
    f = ef.sinh(X)   
    assert f.val == 0.5210953054937474
    assert f.der == 2.2552519304127614
    assert f.jacobian == 1.1276259652063807

def test_sinh_constant_results():
    a = ef.sinh(0.5)
    assert a == 0.5210953054937474

def test_sinh_types():
    with pytest.raises(TypeError):
        ef.sinh('x')
    with pytest.raises(TypeError):
        ef.sinh("1234")
    with pytest.raises(AttributeError):
        ef.sinh({1: 'x'})        

# hyperbolic cos tests
def test_cosh_results():
    X = AutoDiff(0.5, 2)
    f = ef.cosh(X)    
    assert f.val == 1.1276259652063807
    assert f.der == 1.0421906109874948
    assert f.jacobian == 0.5210953054937474
    
def test_cosh_constant_results():
    a = ef.cosh(0.5)
    assert a == 1.1276259652063807 

def test_cosh_types():
    with pytest.raises(TypeError):
        ef.cosh('x')
    with pytest.raises(TypeError):
        ef.cosh("1234")
    with pytest.raises(AttributeError):
        ef.cosh({1: 'x'})       
        
# hyperbolic tan tests
def test_tanh_results():
    X = AutoDiff(0.5, 2)
    f = ef.tanh(X)    
    assert f.val == 0.46211715726000974
    assert f.der == 1.572895465931855
    assert f.jacobian == 0.7864477329659275
    
def test_tanh_constant_results():
    a = ef.tanh(0.5)
    assert a == 0.46211715726000974 

def test_tanh_types():
    with pytest.raises(TypeError):
        ef.tanh('x')
    with pytest.raises(TypeError):
        ef.tanh("1234")
    with pytest.raises(AttributeError):
        ef.tanh({1: 'x'})         

# log10 test
def test_log10_results():
    # value defined at positive real numbers x > 0
    # derivative defined at positive real numbers x > 0
    X = AutoDiff(0.5, 2)
    f = ef.log10(X)    
    assert f.val == -0.3010299956639812
    assert f.der == 1.737177927613007
    assert f.jacobian == 0.8685889638065035
    # neither value nor derivative defined at x <= 0
    with pytest.warns(RuntimeWarning):
        with pytest.raises(TypeError):
            Y = AutoDiff(0, 2)
            f = ef.log10(Y)
    
def test_log10_constant_results():
    a = ef.log10(0.5)
    assert a == -0.3010299956639812

def test_log10_types():
    with pytest.raises(TypeError):
        ef.log10('x')
    with pytest.raises(TypeError):
        ef.log10("1234")
    with pytest.raises(AttributeError):
        ef.log10({1: 'x'})  
