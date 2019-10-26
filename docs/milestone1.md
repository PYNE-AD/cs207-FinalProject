

# AD-PYNE 

Milestone 1

October 29, 2019

# Introduction

Differentiation's use is ubiquitous across the sciences and required for methods such as optimization. Historically, there have existed two ways of computing derivatives: symbolic differentiation and finite differences. Both of these methods have numerous drawbacks. Symbolic differentiation requires hard coding the symbolic derivative for all desired functions. This can be computationally expensive, and it does not address derivatives which are impossible to compute by hand. The method of finite differences can solve these issues, but in turn it requires a good choice of step size for the derivative to be evaluated accurately and in a stable manner. 

Automatic differentiation (AD) is the best of worlds: it is not as computationally expensive, it can address functions whose derivatives are impossible to compute by hand, and it is stable and accurate to machine precision. 

# Background

Automatic differentiation teaches the computer how to calculate derivatives on its own without relying on the human coder's manual coding of derivatives or on potentially unstable and inaccurate approximations. AD breaks down the task of calculating a derivative of any function into series of more simple elementary operations such as addition, multiplication, natural log, etc. 

This is made possible by the Chain Rule. We treat a function as the composition of all of its elementary operations. Thus, the derivative of the original function is the result of the Chain Rule combining the derivatives of is constituents.  

The computational graph of the function allows us to see and record the progress of the Chain Rule at each step of composition. Each elementary function performed gets its own row of current inputs, values, derivative of the elementary function (as determined by previous steps) and the value of the derivative of the elementary function evaluated at the current value. 

In linear algebraic terms, AD computes the derivative as the dot product of the gradient and the seed vector: a vector of derivative values initialized for the variables of interest. 

# How to Use AD-PYNE

The user will interact with the package within a Python script that imports their desired functionality. 

(Different functionalities?)

```python
from AD-PYNE import AutoDiff
```

The user instantiates an `AutoDiff` object by passing in a string function.

```python
myFunction = "5x**2 + sin(3y)"

myAutoDiff = AutoDiff(myFunction)
```

 The user can evaluate the function in their AutoDiff object by calling the `evalFunc` function and passing in an array of values to evaluate at.

```python
myVal = myAutoDiff.evalFunc([3, 4])
```

The result will be a scalar. 

 The user can evaluate the gradient of the function using Forward Mode in their AutoDiff object by calling the `evalGrad` function and passing in an array of values to evaluate at.

```python
myGrad = myAutoDiff.evalGrad([3, 4])
```

The result will be a vector of the partial derivatives. 

# Software Organization 

## Directory Structure 

> AD-PYNE/
>
> ​		AD-PYNE/
>
> ​				\__init__.py
>
> ​				funcReader.py
>
> ​				elemFunctions.py
>
> ​				AutoDiff.py
>
> ​				tests/
>
> ​						funcReader_tests.py
>
> ​						elemFunctions_tests.py
>
> ​						AutoDiff_tests.py
>
> ​		README.md
>
> ​		LICENSE
>
> ​		setup.py
>
> ​		requirements.txt



## Modules

### funcReader

This module parses the string function the user passes into the `AutoDiff` object and returns the relevant elementary functions and input and output size information. 

###  elemFunctions

This module contains the hard-coded derivatives of the elementary functions such as sin, cosine, square root, log, exp, etc. 

### AutoDiff

This module contains the AutoDiff class that is calculates the derivative of a function at a given point using forward mode. 



## Test Suite

The test suite lives in the `tests/` folder. Each module has its own test suite. 

We will be using `TravisCI` to ensure that all tests pass. We will be using `CodeCov` to ensure that all code is covered by a test. 



## Distribution

Distribution will be done through `PyPI` and `twine` will be used to upload the distribution package.



## Packaging 

Software packaging will be done



# Forward Mode Implementation

## Core Data Structures

The core data structures are:

* strings: For passing in a function and parsing the elementary functions.
* `numpy` arrays: For dealing with vector functions of vectors and scalar functions of vectors. 

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
  * `val` is the current value of the function.
  * `der`is the current value of the derivative of the function.



## External Dependencies

### numpy

`numpy` will be used to handle vector operations and math functions such as `sqrt`. 



## Elementary Functions

Elementary functions will be dealt with in a specific module called `elemFunctions.py` which will hold Python functions that return the hard-coded derivative the elementary function. 





