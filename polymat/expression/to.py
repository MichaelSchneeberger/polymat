from numpy.typing import NDArray
import sympy

from statemonad.typing import StateMonad

from polymat.symbols.symbol import Symbol
from polymat.state.state import State as BaseState
from polymat.arrayrepr.arrayrepr import ArrayRepr
from polymat.expressiontree.to import (
    to_array as _to_array,
    to_degree as _to_degree,
    to_numpy as _to_numpy,
    to_shape as _to_shape,
    to_sparse_repr as _to_sparse_repr,
    to_sympy as _to_sympy,
    to_tuple as _to_tuple,
    to_symbols as _to_symbols,
    to_variable_indices as _to_variable_indices,
)
from polymat.expression.typedexpressions import (
    MatrixExpression,
    VariableVectorExpression,
)


def to_array[State: BaseState](
    expr: MatrixExpression[State],
    variables: VariableVectorExpression[State] | tuple[int, ...],
    name: str | None = None,
) -> StateMonad[State, ArrayRepr]:
    return _to_array(expr.child, variables, name=name)


def to_degree[State: BaseState](
    expr: MatrixExpression[State],
    variables: VariableVectorExpression[State] | tuple[int, ...] | None = None,
) -> StateMonad[State, NDArray]:
    return _to_degree(expr.child, variables)


def to_numpy[State: BaseState](
    expr: MatrixExpression[State],
) -> StateMonad[State, NDArray]:
    return _to_numpy(expr.child)


def to_shape[State: BaseState](
    expr: MatrixExpression[State],
) -> StateMonad[State, tuple[int, int]]:
    return _to_shape(expr.child)


def to_sparse_repr[State: BaseState](
    expr: MatrixExpression[State],
):
    return _to_sparse_repr(expr.child)


def to_sympy[State: BaseState](
    expr: MatrixExpression[State],
) -> StateMonad[State, sympy.Expr]:
    return _to_sympy(expr.child)


def to_tuple[State: BaseState](
    expr: MatrixExpression[State], assert_constant: bool = True
) -> StateMonad[State, tuple[tuple[float, ...], ...]]:
    return _to_tuple(expr.child, assert_constant=assert_constant)


def to_symbols[State: BaseState](
    expr: MatrixExpression[State],
) -> StateMonad[State, tuple[Symbol, ...]]:
    return _to_symbols(expr.child)


def to_variable_indices[State: BaseState](
    expr: MatrixExpression[State],
) -> StateMonad[State, tuple[int, ...]]:
    return _to_variable_indices(expr.child)
