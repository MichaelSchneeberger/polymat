# Multivariate Polynomial Library

PolyMat is a Python library designed for the representation and manipulation of multivariate polynomial matrices.


## Features

* Expression Building: Create polynomial expressions using various operators provided by the library.
* Efficient Internal Representation: Uses a sparse internal structure to optimize intermediate computations.
* Deferred Computation: Polynomial matrices are evaluated lazily using the [statemonad](https://github.com/MichaelSchneeberger/state-monad) library, meaning expressions are computed only when needed.
* Finalization: To obtain a concrete representation of the polynomial matrix, use one of the `to_` methods.


## Installation

You can install PolyMat via pip:

```
pip install polymat
```


## Basic Usage

In this example, we define a polynomial expressions using the `+` and `*` operators:

$f(x_1, x_2) = (x_1 + x_2) + (x_1 + x_1 x_2)$

Finally, different representations of the polynomial are printed.

``` python
import polymat

# Initialize state
state = polymat.init_state()

# Define polynomial variables and stack them into a vector
names = ('x1', 'x2')
x1, x2 = (polymat.define_variable(n) for n in names)
x = polymat.v_stack((x1, x2))

# Create a polynomial expression using arithmetic operations
f = (x1 + x2) + (x1 + x1*x2)

# Print a human-readable string representation
print(f'{f}')

# Print the internal Python representation of the expression
print(f'{f=}')

# sympy representation
state, sympy_repr = polymat.to_sympy(f).apply(state)
print(f'{sympy_repr}')

# array representation
state, array_repr = polymat.to_array(f, x).apply(state)
print(f'{array_repr.data[1]=}')   # Dense numpy array
print(f'{array_repr.data[2].toarray()=}')  # Sparse scipy array converted to numpy
```


## Operations

### Creating Polynomial Expressions

- **Polynomial Variable**: Define a polynomial variable with the `polymat.define_variable` function.
- **From Data**: Create a polynomial expression using the `polymat.from_` function from:
    - Tuple of numbers and polynomial variables
    - `numpy` arrays
    - `sympy` expressions (symbols are automatically converted to polynomial variables).

### Combining Polynomial Expressions

- **Block Diagonal**: Combine expression into block diagonal matrices with the `polymat.block_diag` function.
- **Product**: Form vectors containing the Cartesian product of polynomial expressions using the `polymat.product` function.
- **Vertical Stacking**: Stack multiple polynomial expressions vertically using the `polymat.v_stack` function.

### Polynomial Expression Manipulation

- **Arithmetic operations**: Perform addition (`+`), subtraction (`-`), scalar multiplication and division (`*`, `/`), matrix multiplication (`@`), and exponentiation (`**`).
- **Caching**: Cache the `cache` method to store intermediate results and speed up computation.
- [**Combinations**](https://github.com/MichaelSchneeberger/polymat/blob/main/polymat/expressiontree/operations/combinations.py): Stack combinations of elements from polynomial vectors using `combination`.
- **Diagonalization**: Extract or construct diagonal polynomial matrices with `diag`.
- **Differentiation**: Compute derivatives using `diff`.
- **Evaluation**: Replace variables within floats using `eval`.
- **Kronecker Product**: Compute the Kronecker products using `kron`.
- **Linear Coefficient Vector**: Expand a polynomial vector into monomial components using `to_linear_coefficients`.
- **Monomials Terms**: Collect monomials terms using `to_linear_monomials`.
- [**Quadratic Coefficient Matrix**](https://github.com/MichaelSchneeberger/polymat/blob/main/polymat/expressiontree/operations/quadraticcoefficients.py): Compute the Coefficient matrix $Q$ appearing in the quadratic form of the polynomial $p(x) = Z(x)^\top Q Z(x)$ using `to_gram_matrix`.
- [**Quadratic Monomial Terms**](https://github.com/MichaelSchneeberger/polymat/blob/main/polymat/expressiontree/operations/quadraticmonomials.py): Construct a monomial vector $Z(x)$ for the quadratic form of the polynomial $p(x) = Z(x)^\top Q Z(x)$ with `to_quadratic_monomials`.
- **Repmat**: Repeat matrices with `rep_mat`.
- **Reshape**: Modify the shape of polynomial matrices using `reshape`.
- **Summation**: Sum the rows of the polynomial expression using `sum`.

### Output

- **Array Representation**: Convert polynomial expressions to an array representation (implemented through numpy and scipy array) using the `polymat.to_array` function.
- **Tuple Representation**: Outputs constant parts as nested tuple using `polymat.to_tuple`.
- **Sympy Representation**: Convert experssion to `sympy` representation using `polymat.to_sympy`.
- **Polynomial Degrees**: Obtain degrees of each polynomial matrix element using `polymat.to_degree`.
- **Shape of the Matrix**: Retrieve the shape of the polynomial matrix using `polymat.to_shape`.

<!-- 
## References

Here are some references related to this probject:
*  -->
