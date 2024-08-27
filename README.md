# Multivariate Polynomial Library

PolyMat is a library designed to represent and manipulate multivariate polynomial matrices.


## Features

* Expression Building: You can build polynomial expressions using various operators provided by the library.
- Internal representation: An internal (sparse) representation is used to save intermediate results.
- Finalization: To compute an actual representation of the polynomial matrix, call one of the `to_` methods listed below. 
* Deferred Computation: Polynomial matrices are evaluated lazily using the [statemonad](https://github.com/MichaelSchneeberger/state-monad) library. This means that the internal representation of polynomial expressions are only computed when necessary.


## Installation

You can install PolyMat using pip:

```
pip install polymat
```


## Basic Usage

In this example, two polynomial expressions are defined using `sympy` expressions

$f_1(x_1, x_2) = x_1 + x_2$

$f_2(x_1, x_2) = x_1 + x_1 x_2$

Then, the two expression are combined using the `__add__` (or equivalently `+` infix) method

$f_3(x_1, x_2) = f_1(x_1, x_2) + f_2(x_1, x_2) = 2 x_1 + x_2 + x_1 x_2$

Finally, different representations of the polynomial are printed.

``` python
import polymat

# create the state object
state = polymat.init_state()


# (scalar) polynomial matrix expression
#######################################

names = ('x1', 'x2')
x1, x2 = (polymat.define_variable(n) for n in names)
x = polymat.v_stack((x1, x2))

f1 = x1 + x2
f2 = x1 + x1*x2

f3 = f1 + f2

# prints a nicely printable string representation of the expression
# add(add(x1, x2), add(x1, mul(x1, x2)))
print(f'{f3}')

# prints the string representation of the dataclass
# f3 = ExpressionImpl(
#   child=AdditionExprImpl(
#       left=AdditionExprImpl(
#           left=FromVariableImpl(variable='x1', size=1),
#           right=FromVariableImpl(variable='x2', size=1)),
#       right=AdditionExprImpl(
#           left=FromVariableImpl(variable='x1', size=1),
#           right=ElementwiseMultImpl(
#               left=FromVariableImpl(variable='x1', size=1),
#               right=FromVariableImpl(variable='x2', size=1)))))
print(f'{f3=}')


# sympy representation
######################

# computes the sympy representation of the expression
state, sympy_repr = polymat.to_sympy(f3,).apply(state)

# prints the sympy representation
# x1*x2 + 2.0*x1 + x2
print(f'{sympy_repr}')


# array representation
######################

# computes the array representation of the expression
state, array_repr = polymat.to_array(f3, x).apply(state)

# prints the array representations
# array_repr.data[1]=array([[2., 1.]])
print(f'{array_repr.data[1]=}')               # numpy array
# array_repr.data[2].toarray()=array([[0. , 0.5, 0.5, 0. ]])
print(f'{array_repr.data[2].toarray()=}')     # sparse scipy array converted to an numpy array
```


## Operations

### Creating a Polynomial Expressions

- **Polynomial variable**: Creates a polynomial variable using the `polymat.define_variable` function.
- **From data**: Create a polynomial expression using the `polymat.from_` function from:
    - a tuple of numbers and polynomial variables
    - a `numpy` array
    - a `sympy` expression (`sympy` symbols are automatically converted to polynomial variables).

### Combining Polynomial Expressions

- **Block Diagonal**: Create block diagonal matrices of polynomial expressions with the `polymat.block_diag` function.
<!-- - **Horizonal Stacking**: Create multiple polynomial expressions horizontally using `polymat.h_stack`. -->
- **Product**: Create a vector containing all elements of the Cartesian product of multiple polynomial expressions using `polymat.product`.
- **Vertical Stacking**: Combine multiple polynomial expressions vertically using `polymat.v_stack`.

### Polynomial Expression Manipulation

- **Arithmetic operations**: Compute addition, subtraction, scalar multiplication, scalar division and matrix multiplication using the `+`, `-`, `*`, `/`, `@`, and `**` methods.
- **Caching**: Cache the intermediate representation of the polynomial expression to spead up computation using the `cache` method.
- **Combinations**: Compute all combinations of the elements of a polynomial vector up to a given order and stack the result into a polynomial vector using the `combination` method.
- **Diagonalization**: Extract a diagonal or construct a diagonal polynomial matrix using the `diag` method.
- **Differentiation**: Compute derivatives using the `diff` method.
- **Evaluation**: Replace variables within the polynomial expression by a float using the `eval` method.
<!-- - **Filter vector**: -->
- **Kronecker Product**: Compute the Kronecker product using the `kron` method.
- **Linear Expansion**: Expand a polynomial vector into monomial components given by a monomial vector using the `linear_in` method.
- **Monomials Terms**: Collect monomials terms of the polynomial expression using the `linear_monomials_in` method.
- **Quadratic Expansion**: Compute the Gram matrix of a polynomial expression using the `quadratic_in` method.
- **Quadratic Monomial Terms**: Computes a monomial vector of the quadratic form of a polynomial expression using the `quadratic_monomials_in` method.
- **Repmat**: Repeat a matrix $m, n$ times using the `rep_mat` method.
- **Reshape**: Modify the shape of polynomial matrices with the `reshape` method.
- **Summation**: Sum rows of the polynomial expression using the `sum` method.

### Output

- **Array Representation**: Convert polynomial expressions to an array representation (implemented through numpy and sympy array) using the `polymat.to_array` function.
- **Tuple Representation**: Outputs the constant part of a polynomial expression into a nested tuple of float using `polymat.to_tuple`.
- **Sympy Representation**: Obtain a sympy representation of the polynomial expression using the `polymat.to_sympy` method.
- **Polynomial Degrees**: Obtain the polynomial degree of each element in the matrix using the `polymat.to_degree` method.
- **Shape of the Matrix**: Obtain the shape of the polynomial matrix using the `polymat.to_shape` method.
