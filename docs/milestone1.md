

# AD-PYNE 

###### A(utomatic) D(ifferentiation) - (As developed by) P(aulina) Y(aowei) N(ikhil) E(mma)

Milestone 1

October 29, 2019

# Introduction

Differentiation's use is ubiquitous across the sciences and required for methods such as optimization. Historically, there have existed two ways of computing derivatives: symbolic differentiation and finite differences. Both of these methods have numerous drawbacks. Symbolic differentiation requires hard coding the symbolic derivative for all desired functions. This can be computationally expensive, and it does not address derivatives which are impossible to compute by hand. The method of finite differences can solve these issues, but in turn it requires a good choice of step size for the derivative to be evaluated accurately and in a stable manner. 

Automatic differentiation (AD) is the best of worlds: it is not as computationally expensive, it can address functions whose derivatives are impossible to compute by hand, and it is stable and accurate to machine precision. 

# Background

Automatic differentiation teaches the computer how to calculate derivatives on its own without relying on the human coder's manual coding of derivatives or on potentially unstable and inaccurate approximations. AD breaks down the task of calculating a derivative of any function into a series of more simple elementary operations such as addition, multiplication, powers, natural log, etc. 

### Chain Rule

This breakdown is made possible by the Chain Rule, where F(x) is a composition of two or more functions.
$$
F(x) = f(g(x))
$$

$$
F'(x) = f'(g(x))g'(x)
$$

In automatic differentiation, we treat a function as the composition of all of its elementary operations. Thus, the derivative of the original function is the result of the Chain Rule combining the derivatives of is constituents.  

### Computational Graph

The computational graph of the function allows us to see and record the progress of the Chain Rule at each step of composition. 
$$
F(x) = 3x^2 + sin(x)
$$
In the graph above, we can see the input term on the left side. We break the function into nodes, each representing an elementary function performed on previous nodes. The computational graph gives us a visual understanding of how the function is being built up and how we are applying the Chain Rule to form the derivative.

The computational graph visualizes the trace table where each elementary function performed gets its own row of current inputs, values, the derivative of the elementary function (as determined by previous steps) and the value of the derivative of the elementary function evaluated at the current value. 

In linear algebraic terms, AD computes the derivative as the dot product of the gradient and the seed vector: a vector of derivative values initialized for the variables of interest. 

# How to Use AD-PYNE

The user will interact with the package within a Python script that imports their desired functionality. 

```python
from AD-PYNE.AutoDiff import AutoDiff
from AD-PYNE.elemFunctions import *
```

The user instantiates an empty `AutoDiff` object. The user will build up their function using elementary functions and this empty `AutoDiff`object. 

```python
# User instantiates empty AutoDiff object
x = AutoDiff()

# User builds up function
f = 3*x**2 + sin(x)
```

 The user can evaluate the function in their `AutoDiff` object by calling the `evalFunc` function and passing in an array of values to evaluate at. (See the  **Data Structures** section for evaluating at a single point.) `evalFunc` will set the `.val`attribute of the `AutoDiff` object to the value of the function at the point (or vector) passed in.

```python
# Evaluate the function at the desired points
f.evalFunc([[3, 4]])

# Access the evaluated function
myVal = f.val
```

The result will be a scalar or vector depending on the shape of the input. 

 The user can evaluate the gradient of the function using Forward Mode in their AutoDiff object by calling the `evalGrad` function and passing in an array of values to evaluate at. `evalGrad` will set the `.der`arribute of the AutoDiff object to the value of the gradient at the point (or vector) passed in.

```python
# Evaluate the derivative of the function at the desired points
f.evalGrad([3, 4])

# Access the evaluated derivative
myDerivative = f.der
```

The result will be a vector of the evaluated partial derivatives. 

### Justification for Approach

We chose to build up the `AutoDiff` objects first and evaluating later because it would allow the user to first define the general function without considering specific evaluation points. We expect this might be more involved than evaluating the functions as we go along, but we believe this will relieve the user from having to recreate their full function every time they want to evaluate at a different point. 

# Software Organization 

## Directory Structure 


	AD-PYNE/

	    AD-PYNE/

		\__init__.py

		elemFunctions.py

		AutoDiff.py

		tests/

		    elemFunctions_tests.py

		    AutoDiff_tests.py

		README.md

		LICENSE

		setup.py

		requirements.txt



## Modules

###  elemFunctions

This module contains the hard-coded derivatives of the elementary functions such as sin, cosine, square root, log, exp, etc. Thus, we are creating our own custom elementary math functions using the `numpy` math functions that can be performed on  `AutoDiff` objects and will return` AutoDiff` objects. Duck typing will allow the user to pass in (vectors) of scalars and return (vectors) of scalars. 

### AutoDiff

This module contains the `AutoDiff` class that calculates the derivative of a function at a given point using forward mode. It will import the functions from the **elemFunctions** module to use in the class functions `evalFunc` and `evalGrad`.



## Test Suite

The test suite lives in the `tests/` folder. Each module has its own test suite. 

We will be using `TravisCI` to ensure that all tests pass. We will be using `CodeCov` to ensure that all code is covered by a test. 



## Distribution

Distribution will be done through `PyPI` and `twine` will be used to upload the distribution package.



## Packaging 

Software packaging will be done without a framework and through Python's native distribution tools. Our target audience is the developer (or developer-student) audience. We are only packaging a library that will be used solely within Python by users who are assumed to know how to install and use Python packages. It is not a full fledged application like a web application nor is it its own executable software. The software organization is minimal and can be handled manually. 



# Forward Mode Implementation

## Core Data Structures

The core data structures are:

* `numpy` arrays: We will be treating all values and arguments as array. 
  * A single variable, integer, or float will be treated as a 1 x 1 `numpy` array. This is done so that we don't have to deal with single values separately from vectors. 
  * Lists will be converted to 1 x n `numpy` arrays.
  * Users will not be allowed to pass in dictionaries to set their values. The evaluated output and derivative of the `AutoDiff` object will stored in their respective attributes and not in a general `params` attribute that holds a dictionary in the form `{'val': [], 'der': [] }`.

## Classes: Methods, and Name Attributes 

### AutoDiff

* Methods
  * `evalFunc` evaluates the function at a given vector of values. 
  * `evalGrad` evaluates the gradient at a given vector of values. 
  * `__add__`
  * `__radd__`
  * `__mul__`
  * `__rmul__`
* Attributes
  * `val` is the value of the function.
  * `der`is the value of the derivative of the function.



## External Dependencies

### numpy

`numpy` will be used to handle vector operations and math functions such as `sqrt`. 



## Elementary Functions

Elementary functions will be dealt with in a specific module called `elemFunctions.py` which will hold Python functions that return the hard-coded derivative the elementary function. 

An example for `sin` is illustrated below.

```python
def sin(x):
	''' Compute the sin of an AutoDiff object and its derivative.
	
	INPUTS
	======
	x: an AutoDiff object
	
	RETURNS
	=======
	A new AutoDiff object with calculated value and derivative.
	
	EXAMPLES
	========
	>>> sinAutoDiff = sin(x)
	>>> sinAutoDiff.val
	np.sin(x.val)
	>>> sinAutoDiff.der
	np.cos(x.val)*x.der
	
	'''
    new_x = AutoDiff()
    	try:
            new_x.val = np.sin(x.val)
            new_x.der = np.cos(x.val)*x.der
            return new_x
        except AttributeError:
            return np.sin(x)


    
```



