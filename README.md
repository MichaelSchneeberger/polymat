# Multivariate Polynomial Library

PolyMat is a Python library designed for the representation and manipulation of multivariate polynomial matrices.


## Features

* Expression Building: Create polynomial expressions using various operators provided by the library.
* Efficient Internal Representation: Uses a sparse internal structure to optimize intermediate computations.
* Deferred Computation: Polynomial matrices are evaluated lazily using the [statemonad](https://github.com/MichaelSchneeberger/state-monad) library, meaning expressions are computed only when needed.
* Performance Optimized: Designed for speed, the library outperforms other symbolic computation tools like `sympy`, making it ideal for large matrix expressions.


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

- **Polynomial Variable**: Define a polynomial variable.
    ``` python
    x = polymat.define_variable('x')
    ```
- **From Data**: Create a polynomial expression from:
    - Tuple of numbers and polynomial variables
        ``` python
        j = polymat.from_(((0, -1), (1, 0)))
        # Matrix([[0, -1], [1, 0]])
        ```
    - `numpy` arrays (possibly containing polynomial variables)
        ``` python
        i = polymat.from_(np.eye(2))
        # Matrix([[1, 0], [0, 1]])
        ```
    - `sympy` expressions (symbols are automatically converted to polynomial variables).

### Combining Polynomial Expressions

- **Block Diagonal**: Combine expression into block diagonal matrices.
    ``` python
    xblk = polymat.block_diag((x, x))
    # Matrix([[x, 0], [0, x]])
    ```
- **Horizontal Stacking**: Stack multiple polynomial expressions horizontally.
    ``` python
    xhstack = polymat.h_stack((x, x))
    # Matrix([[x, x]])
    ```
<!-- - **Product**: Form vectors containing the Cartesian product of polynomial expressions using the `polymat.product` function.
    ``` python
    xhstack = polymat.h_stack((x, x))
    # Matrix([[x, x]])
    ``` -->
- **Vertical Stacking**: Stack multiple polynomial expressions vertically.
    ``` python
    xvstack = polymat.v_stack((x, x))
    # Matrix([[x], [x]])
    ```

### Polynomial Expression Manipulation

- **Arithmetic operations**: Perform addition (`+`), subtraction (`-`), scalar multiplication and division (`*`, `/`), matrix multiplication (`@`), and exponentiation (`**`).
    ``` python
    f = j * x**2 - i
    # Matrix([[-1.0, -1.0*x**2], [x**2, -1.0]])
    ```
- **Caching**: Cache the polynomial expression to store intermediate results and speed up computation.
    ``` python
    x = x.cache()
    ```
- [**Combinations**](https://github.com/MichaelSchneeberger/polymat/blob/main/polymat/expressiontree/operations/combinations.py): Stack multiplied combinations of elements from polynomial vectors.
    ``` python
    m = x.combinations((0, 1, 2))
    # Matrix([[1], [x], [x**2]])
    ```
- **Diagonalization**: Extract a diagonal or construct diagonal matrix.
    ``` python
    mdiag = m.diag()
    # Matrix([[1, 0, 0], [0, x, 0], [0, 0, x**2]])
    ```
- **Differentiation**: Compute the Jacobian matrix of a polynomial vector.
    ``` python
    mdiff = m.diff(x)
    # Matrix([[0], [1], [2.0*x]])
    ```
- **Evaluation**: Replace variable symbols within tuple of floats.
    ``` python
    meval = m.eval({x.symbol: (2,)})
    # Matrix([[1], [2.0], [4.0]])
    ```
- **Kronecker Product**: Compute the Kronecker products.
    ``` python
    fkron = f.kron(i)
    # Matrix([[-1.0, 0, -1.0*x**2, 0], [0, -1.0, 0, -1.0*x**2], [x**2, 0, -1.0, 0], [0, x**2, 0, -1.0]])
    ```
- **Repmat**: Repeat polynomial expressions.
    ``` python
    xrepmat = x.repmat(3, 1)
    Matrix([[x], [x], [x]])
    ```
- **Reshape**: Modify the shape of polynomial matrices.
    ``` python
    freshape = f.reshape(-1, 1)
    # Matrix([[-1.0], [x**2], [-1.0*x**2], [-1.0]])
    ```
- **Summation**: Sum the rows of the polynomial expression.
    ``` python
    fsum = f.sum()
    # Matrix([[-1.0*x**2 - 1.0], [x**2 - 1.0]])
    ```

Specialized methods:
- **Monomials Terms**: Construct a monomial vector $Z(x)$ appearing in a polynomial expression.
    ``` python
    p = x**3 - 2*x + 3

    p_monom = p.to_linear_monomials(x)
    # Matrix([[1], [x], [x**3]])
    ```
- **Linear Coefficient Vector**: Compute a coefficient matrix $Q$ associated with a vector of monomials $Z(x)$ and a polynomial vector $p(x) = Q Z(x)$.
    ``` python
    p_coeff = p.to_linear_coefficients(x, monomials=p_monom)
    # Matrix([[3, -2.0, 1]])
    ```
- [**Quadratic Monomial Terms**](https://github.com/MichaelSchneeberger/polymat/blob/main/polymat/expressiontree/operations/quadraticmonomials.py): Construct a monomial vector $Z(x)$ for the quadratic form of the polynomial $p(x) = Z(x)^\top Q Z(x)$.
    ``` python
    p_monom = p.to_quadratic_monomials(x)
    # Matrix([[1], [x], [x**2]])
    ```
- [**Quadratic Coefficient Matrix**](https://github.com/MichaelSchneeberger/polymat/blob/main/polymat/expressiontree/operations/quadraticcoefficients.py): Compute the symmetric coefficient matrix $Q$ appearing in the quadratic form of the polynomial $p(x) = Z(x)^\top Q Z(x)$.
    ``` python
    p_coeff = p.to_gram_matrix(x, monomials=p_monom)
    # Matrix([[3.0, -1.0, 0], [-1.0, 0, 0.5], [0, 0.5, 0]])
    ```




### Output

The output functions listed below perform stateful computations. As a result, they return a StateMonad object, which must be applied with a state object to generate the desired output value.

- **Sympy Representation**: Convert an experssion to a `sympy` representation.
    ``` python
    state, sympy_repr = polymat.to_sympy(f).apply(state)
    # Matrix([[-1.0, -1.0*x**2], [x**2, -1.0]])
    ```
- **Array Representation**: Convert polynomial expressions to an array representation (implemented through numpy and scipy array)..
    ``` python
    state, farray = polymat.to_array(f, x).apply(state)
    # {0: array([[-1.], [ 0.], [ 0.], [-1.]]), 2: array([[ 0.], [ 1.], [-1.], [ 0.]])}
    ```
- **Tuple Representation**: Outputs constant parts as nested tuple.
    ``` python
    # Setting assert_constant=False will prevent an exception form being raised, even if f is not a constant polynomial expression
    state, ftuple = polymat.to_tuple(f, assert_constant=False).apply(state)
    # ((-1.0,), (-1.0,))
    ```
- **Polynomial Degrees**: Obtain degrees of each polynomial matrix element.
    ``` python
    state, fdegree = polymat.to_degree(f).apply(state)
    # (2, 2)
    ```
- **Shape of the Matrix**: Retrieve the shape of the polynomial matrix.
    ``` python
    state, fshape = polymat.to_shape(f).apply(state)
    # ((0, 2), (2, 0))
    ```


## References

Here are some references related to this probject:

* [State-Monad](https://github.com/MichaelSchneeberger/state-monad) is a Python library that encapsulates stateful computations into a monadic structure.