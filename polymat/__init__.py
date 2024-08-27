"""
The root modules of polymat are meant to be self-containing and could be placed in a package of its own.
They all contain an abc.py exposing the interfaces that can be extended by an external library.
"""

from polymat.state import (
    init_state as _init_state,
)
from polymat.expression.from_ import (
    from_ as _from_,
    from_symmetric as _from_symmetric,
    from_vector as _from_vector,
    from_row_vector as _from_row_vector,
    from_polynomial as _from_polynomial,
    from_variables as _from_variables,
    from_variable_indices as _from_variable_indices,
    define_variable as _define_variable,
    block_diag as _block_diag,
    concat as _concat,
    h_stack as _h_stack,
    product as _product,
    v_stack as _v_stack,
)
from polymat.expression.to import (
    to_array as _to_array,
    to_degree as _to_degree,
    to_numpy as _to_numpy,
    to_shape as _to_shape,
    to_sparse_repr as _to_sparse_repr,
    to_sympy as _to_sympy,
    to_tuple as _to_tuple,
    to_variables as _to_variables,
    to_variable_indices as _to_variable_indices,
)

init_state = _init_state

from_ = _from_
from_symmetric = _from_symmetric
from_vector = _from_vector
from_row_vector = _from_row_vector
from_polynomial = _from_polynomial
# from_variables = _from_variables              # remove?
from_variable_indices = _from_variable_indices
define_variable = _define_variable

block_diag = _block_diag
concat = _concat
h_stack = _h_stack
product = _product
v_stack = _v_stack

to_array = _to_array
to_degree = _to_degree
to_numpy = _to_numpy
to_shape = _to_shape
to_sparse_repr = _to_sparse_repr
to_sympy = _to_sympy
to_tuple = _to_tuple
# to_variables = _to_variables              # remove?
to_variable_indices = _to_variable_indices
