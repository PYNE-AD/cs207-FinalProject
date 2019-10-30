

# AD-PYNE 

###### A(utomatic) D(ifferentiation) - (As developed by) P(aulina) Y(aowei) N(ikhil) E(mma)

Milestone 1

October 29, 2019

# Introduction

Differentiation's use is ubiquitous across the sciences and is required for common methods such as optimization. Historically, there have existed two ways of computing derivatives: symbolic differentiation and method of finite differences. Both of these techniques have numerous drawbacks. Symbolic differentiation involves manipulating the abstract formula using mathematical expressions and rules to produce a desired derivative formula. This can be computationally expensive, and it does not address derivatives which are impossible to compute by hand, and have no "closed-form" solution. The method of finite differences can solve these issues, but in turn it requires a good choice of step size for the derivative to be evaluated accurately and in a stable manner. 

Automatic differentiation (AD) is the best of worlds: it is not as computationally expensive, it can address functions whose derivatives are impossible to compute by hand, and it is stable and accurate to machine precision. 

# Background

Automatic differentiation teaches the computer how to calculate derivatives on its own without relying on computationally expensive symbolic differentiation programs or on potentially unstable and inaccurate approximations with the finite-difference method. AD breaks down the task of calculating a derivative of any function into a series of more simple elementary operations such as addition, multiplication, powers, natural log, etc., whose derivatives we already know. We piece together these derivatives using the Chain Rule.

### Chain Rule

This breakdown is made possible by the Chain Rule, where F(x) is a composition of two or more functions.

![Composition Equation](https://latex.codecogs.com/gif.latex?F%28x%29%20%3D%20f%28g%28x%29%29)

![Derivative of Composition](https://latex.codecogs.com/gif.latex?F%27%28x%29%20%3D%20f%27%28g%28x%29%29g%27%28x%29)

In automatic differentiation, we treat a function as the composition of all of its elementary operations. Thus, the derivative of the original function is the result of using the Chain Rule to combine the derivatives of its constituents.  

### Computational Graph

The computational graph of the function allows us to see and record the progress of the Chain Rule at each step of composition. 

**Example function:**

![Example Function](https://latex.codecogs.com/gif.latex?F%28x%29%20%3D%203x%5E2%20&plus;%20sin%28x%29)

**Example computational graph:**

![Computational Graph](images/comp_graph.png)

In the graph above, we can see the input term on the left side. We break the function into nodes, each representing an elementary function performed on previous nodes. The computational graph gives us a visual understanding of how the function is being built up and how we are applying the Chain Rule to form the final derivative.

### Trace Table

The computational graph visualizes the trace table where each elementary function performed gets its own row of current inputs, values, the derivative of the elementary function (as determined by previous steps) and the value of the derivative of the elementary function evaluated at the current value. 

**Example trace table:**

| Trace         | Elementary Function           | Elementary Function Derivative   | Current Value           | ∇x          |
| ------------- | ----------------------------- | -------------------------------- | ----------------------- | ----------- |
| x<sub>1</sub> | x<sub>1</sub>                 | ẋ<sub>1</sub>                    | x                       | 1           |
| x<sub>2</sub> | x<sub>1</sub><sup>2</sup>     | 2x<sub>2</sub>ẋ<sub>1</sub>      | 2x                      | 2x          |
| x<sub>3</sub> | 3x<sub>2</sub>                | 3ẋ<sub>2</sub>                   | 3x<sup>2</sup>          | 6x          |
| x<sub>4</sub> | sin(x<sub>1</sub>)            | cos(x<sub>1</sub>) ẋ<sub>1</sub> | sin(x)                  | cos(x)      |
| x<sub>5</sub> | x<sub>3</sub> + x<sub>4</sub> | ẋ<sub>3</sub> + ẋ<sub>4</sub>    | 3x<sup>2</sup> + sin(x) | 6x + cos(x) |

Automatic Differentiation moves forward through this graph (but does not necessarily have to record all the rows) to calculate the derivative of the full function. The value of the derivative ẋ<sub>1</sub> is determined by a seed vector: a vector of derivative values initialized for the variables of interest. 

In linear algebraic terms, AD computes the derivative as the dot product of the gradient and the seed vector.

# How to Use AD-PYNE

The user will interact with our package within a Python script that imports their desired functionality. 

```python
from AD-PYNE.AutoDiff import AutoDiff
import AD-PYNE.elemFunctions as adef
```
### Scalar Functions of Vectors

**Note**:  As discussed in the **Data Structures** section below, a single value will be treated as a `numpy` array of size 1 x 1. Thus, <u>scalar functions of scalars</u> are treated as scalar functions of 1 x 1 vectors.

After the user instantiates an empty `AutoDiff` object, they will build up their desired function using elementary functions and this empty `AutoDiff`object. 

```python
# User instantiates empty AutoDiff objects, the default seed for each variable is 1
x = AutoDiff()
y = AutoDiff()

# User builds up function
f = 3*(x**2) + adef.sin(x-y)
```

The user can evaluate the function in their `AutoDiff` object by calling the `evalFunc` function and passing in a row vector of values to evaluate at. `evalFunc` will set the `.val`attribute of the `AutoDiff` object to the value of the function at the vector passed in.

```python
# Evaluate the function at the desired point x = 3 and y = 4
f.evalFunc(np.array([[3, 4]]))

# Access the evaluated function
myVal = f.val
```

The result will be a scalar.

The user can evaluate the gradient of the function using Forward Mode in their AutoDiff object by calling the `evalGrad` function and passing in an array of values to evaluate at. `evalGrad` will set the `.der` attribute of the AutoDiff object to the value of the gradient at the vector passed in.

```python
# Evaluate the derivative of the function at the desired points x = 3 and y = 4
f.evalGrad(np.array([[3, 4]]))

# Access the evaluated derivative
myDerivative = f.der

# The partial derivative of x is stored in the first element of f.der
myDerivativeX = f.der[0]

# The partial derivative of y is stored in the second elmenet of f.der
myDerivativeY = f.der[1]
```

The result will be a row vector of the evaluated partial derivatives. 

### Vector Functions of Vectors

The user can also build up vector functions with vectors. Vector functions of scalars are treated as a vector of size n x 1.

```python
# User instantiates empty AutoDiff objects
X = AutoDiff()
Y = AutoDiff()

# User builds up function
F = 3*(X**2) + adef.sin(X*Y)

# User defines vectors for evaluation
Xs = np.linspace(1, 100, num=100)
Ys = np.linespace(-50, 50, num=100)

# Evaluate the vector functions at the desired points
F.evalFunc(np.array([[Xs], [Ys]]))
myVal = F.val
# Returns a vector size 100 x 1, where the evaluated value of each ith function is found in the ith row

# Evaluate the derivative of the function at the desired points
F.evalFunc(np.array([[Xs], [Ys]]))
myDerivative = F.der
# Returns a vector of size 100 x 2

# The vector partial derivatives of X are stored in the first column of f.der
myDerivativeX = F.der[:, 0]

# The vector derivatives of Y are stored in the second column of f.der
myDerivativeY = F.der[:, 1]
```

### Justification for Approach

We chose to build up the `AutoDiff` objects first and evaluating afterwards because it would allow the user to first define the general function without considering specific evaluation points. We expect this might be more involved than evaluating the functions as we go along, but we believe this will relieve the user from having to recreate their full function every time they want to evaluate at a different point. 

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

This module contains the hard-coded derivatives of the elementary functions such as sine, cosine, square root, log, exp, etc. Thus, we are creating our own custom elementary math functions using `numpy` math functions that can be performed on  `AutoDiff` objects, and will return` AutoDiff` objects. Duck typing will also allow the user to pass in (vectors of) scalars and return (vectors of) scalars. 

### AutoDiff

This module contains the `AutoDiff` class that calculates the derivative of a function at a given point using the forward mode of automatic differentiation. It will import the functions from the **elemFunctions** module to use in the class functions `evalFunc`, `evalGrad`, `evalJacobian`.



## Test Suite

The test suite lives in the `tests/` folder. Each module has its own test suite. 

We will be using `TravisCI` to ensure that all tests pass.
We will be using `CodeCov` to ensure that all code is covered by a test. 



## Distribution

Distribution will be done through `PyPI` and `twine` will be used to upload the distribution package.



## Packaging 

Software packaging will be done without a framework and through Python's native distribution tools. Our target audience is a developer (or developer-student) audience. We are only packaging a library that will be used solely within Python by users who are assumed to know how to install and use Python packages. It is not a full fledged application like a web application nor is it its own executable software. The software organization is minimal and can be handled manually. 



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
  * `evalFunc` evaluates the function at a given vector of values and stores in `self.val` .
  * `evalGrad` evaluates the gradient at a given vector of values and stores in `self.der`.  If `self.seed` is the identity, the function also stores the calculated gradient in `self.jacobian`.
  * `evalJacobian` evaluates the Jacobian matrix (a matrix of all first order partial derivatives) and stores in `self.jacobian`. 
  * Overloaded Python operations:
    * `__add__`
    * `__radd__`
    * `__sub__`
    * `__rsub__`
    * `__mul__`
    * `__rmul__`
    * `__truediv__`
    * `__rtruediv__`
    * `__pow__`
    * `__rpow__`
* Attributes
  * `val` is the value of the function.
  * `der`is the value of the derivative of the function.
  * `seed` is the seed vector, defaulted to the identity. 
  * `jacobian` is the Jacobian matrix, defaulted to `None`. 

## External Dependencies

### numpy

`numpy` will be used to handle vector operations and math functions such as `sqrt`. 

## Elementary Functions

Elementary functions will be dealt with in a specific module called `elemFunctions.py` which will hold Python functions that return the hard-coded derivative the elementary function. Functions will include many of those defined in `numpy`:
* `sin`
* `cos`
* `tan`
* `arcsin`
* `arccos`
* `arctan`
* `sinh`
* `cosh`
* `tanh`
* `arcsinh`
* `arccosh`
* `arctanh`
* `log`
* `log10`

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



