# Multivariate Polynomial Library

PolyMat is a library designed to represent and manipulate multivariate polynomial matrices.


## Installation

You can install PolyMat using pip:

```
pip install polymat
```

<!-- # Multivariate Polynomial Library

`polymat` is a library designed to represent and manipulate multivariate polynomial matrices.

## Key Features

### Lazy Evaluation

- **Deferred Computation**: `polymat` uses lazy evaluation for building polynomial expressions. This means that polynomial expressions are created incrementally and are only fully computed when necessary.
- **Expression Building**: You can build polynomial expressions using various operators provided by the library.
- **Internal representation**: An internal (sparse) representation is used to save intermediate results.
- **Finalization**: To compute an actual representation of the polynomial matrix, call one of the `to_` methods listed below. 

### Creating a Polynomial Expression

- **From numbers**: Create a polynomial expression from numbers using `polymat.from_` function.
- **From numpy**: Create a polynomial expression from numpy expressions using `polymat.from_` function.
- **From sympy**: Create a polynomial expression from sympy expressions using `polymat.from_` function.

### Combining Polynomial Expressions

- **Block Diagonal**: Create block diagonal matrices of polynomial expressions with `polymat.block_diag`.
- **Horizonal Stacking**: Create multiple polynomial expressions horizontally using `polymat.h_stack`.
- **Kronecker Product**: 
- **Product**: Create a vector containing all elements of the Cartesian product of multiple polynomial expressions using `polymat.product`.
- **Vertical Stacking**: Combine multiple polynomial expressions vertically using `polymat.v_stack`.

### Polynomial Expression Manipulation

- **Arithmetic operations**: Compute addition, subtraction, scalar multiplication, scalar division and matrix multiplication using the `__add__`, `__sub__`, `__mul__`, `__truediv__`, and `__matmul__` methods.
- **Caching**: Cache the intermediate representation of the polynomial expression in the state.
- **Combinations**:
- **Diagonalization**:
- **Differentiation**: Compute derivatives using the `diff` method.
- **Evaluation**: Replace variables within expressions using the `eval` method.
- **Filter vector**:
- **Reshape**: Modify the shape of polynomial matrices with the `reshape` method.
- **Summation**: Sum polynomial expressions using the `sum` method.

### Matrix Representation

- **Matrix Conversion**: Convert polynomial expressions to matrix representations using `polymat.to_array`.
- **Evaluation**: To obtain the actual matrix representation, call the `apply(state)` method after conversion. -->



<!-- ## Usage

To get started with `polymat`, you can: -->


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
