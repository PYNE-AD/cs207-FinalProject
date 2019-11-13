## Jacobian-vector product
The Trace Table above shows us how to manually perform the forward mode of automatic differentiation for scalar functions of a single variable, however in the real life we need to tackle systems of equations ($f$ becomes a vector), which reqiures differentation of a vector-function of multiple variables. On important element of this process is the Jacobian matrix, wihc is mainly the matrix of partial derivatives

$$
\mathbf{J}=\left[\begin{array}{ccc}{\frac{\partial \mathbf{f}}{\partial x_{1}}} & {\cdots} & {\frac{\partial \mathbf{f}}{\partial x_{n}}}\end{array}\right]=\left[\begin{array}{ccc}{\frac{\partial f_{1}}{\partial x_{1}}} & {\cdots} & {\frac{\partial f_{1}}{\partial x_{n}}} \\ {\vdots} & {\ddots} & {\vdots} \\ {\frac{\partial f_{m}}{\partial x_{1}}} & {\cdots} & {\frac{\partial f_{m}}{\partial x_{n}}}\end{array}\right]
$$

We know from above that the AD computes the derivative as the dot product of the gradient and the seed vector, which can be wirten as $\nabla f \cdot p$. If we consider the Jacobian form, what orward mode actually computes is $J p$. What's more, we can choose the value of the seed vectors $\left\{p_{1}, \dots p_{n}\right\}$ where $p_{i} \in \mathbb{R}^{n}$ to the entire or part of the Jacobian depending on our applications.
