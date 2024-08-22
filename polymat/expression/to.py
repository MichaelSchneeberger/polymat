from numpy.typing import NDArray
import sympy

from statemonad.typing import StateMonad

from polymat.arrayrepr.abc import ArrayRepr
from polymat.variable import Variable
from polymat.state import State
from polymat.expressiontree.to import (
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
from polymat.expression.abc import Expression, VariableVectorExpression


def to_array(
    expr: Expression,
    variables: VariableVectorExpression,
) -> StateMonad[State, ArrayRepr]:
    return _to_array(expr.child, variables)


def to_degree(
    expr: Expression,
    variables: VariableVectorExpression | None = None,
) -> StateMonad[State, NDArray]:
    return _to_degree(expr.child, variables)


def to_numpy(expr: Expression) -> StateMonad[State, NDArray]:
    return _to_numpy(expr.child)


def to_shape(expr: Expression) -> StateMonad[State, tuple[int, int]]:
    return _to_shape(expr.child)


def to_sparse_repr(expr: Expression):
    return _to_sparse_repr(expr.child)


def to_sympy(expr: Expression) -> StateMonad[State, sympy.Expr]:
    return _to_sympy(expr.child)


def to_tuple(expr: Expression) -> StateMonad[State, tuple[tuple[float, ...], ...]]:
    return _to_tuple(expr.child)


def to_variables(expr: Expression) -> StateMonad[State, tuple[Variable, ...]]:
    return _to_variables(expr.child)


def to_variable_indices(
    expr: VariableVectorExpression,
) -> StateMonad[State, tuple[int, ...]]:
    """
    Convert a variable vector expression into a tuple of variable indices.

    This function iterates over the rows of the provided `VariableVectorExpression`
    and collects the indices corresponding to the variable at each row of the vector.
    The resulting indices are aggregated into a tuple, which is returned within a
    `StateMonad`.

    Example:
    ``` python
    x1, x2, x3 = (polymat.define_variable(name) for name in range(3))
    x = polymat.v_stack((x1, x2, x3))
    state, indices = polymat.to_variable_indcies(x).apply(state)

    print(indices)  # Output will be (0, 1, 2)
    ```

    Args:
        expr (VariableVectorExpression): The variable vector expression containing
        the rows to be iterated over.

    Returns:
        StateMonad[State, tuple[int, ...]]: A stateful monad containing the tuple
        of variable indices for the expression's rows.
    """

    return _to_variable_indices(expr.child)
