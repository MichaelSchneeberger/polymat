from numpy.typing import NDArray
import sympy

from statemonad.typing import StateMonad

from polymat.arrayrepr.arrayrepr import ArrayRepr
from polymat.symbol import Symbol
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
from polymat.expression.typedexpressions import MatrixExpression, VariableVectorExpression


def to_array(
    expr: MatrixExpression,
    variables: VariableVectorExpression | tuple[int, ...],
    name: str | None = None,
) -> StateMonad[State, ArrayRepr]:
    return _to_array(expr.child, variables, name=name)


def to_degree(
    expr: MatrixExpression,
    variables: VariableVectorExpression | None = None,
) -> StateMonad[State, NDArray]:
    return _to_degree(expr.child, variables)


def to_numpy(expr: MatrixExpression) -> StateMonad[State, NDArray]:
    return _to_numpy(expr.child)


def to_shape(expr: MatrixExpression) -> StateMonad[State, tuple[int, int]]:
    return _to_shape(expr.child)


def to_sparse_repr(expr: MatrixExpression):
    return _to_sparse_repr(expr.child)


def to_sympy(expr: MatrixExpression) -> StateMonad[State, sympy.Expr]:
    return _to_sympy(expr.child)


def to_tuple(expr: MatrixExpression) -> StateMonad[State, tuple[tuple[float, ...], ...]]:
    return _to_tuple(expr.child)


def to_variables(expr: MatrixExpression) -> StateMonad[State, tuple[Symbol, ...]]:
    return _to_variables(expr.child)


def to_variable_indices(
    expr: MatrixExpression,
) -> StateMonad[State, tuple[int, ...]]:
    return _to_variable_indices(expr.child)
