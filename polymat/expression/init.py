from typing import override
from dataclassabc import dataclassabc

from polymat.expression.abc import (
    MatrixExpression,
    VariableExpression,
)
from polymat.expressiontree.expressiontree import ExpressionTree
from polymat.symbol import Symbol


@dataclassabc(frozen=True)
class ExpressionImpl(MatrixExpression):
    child: ExpressionTree

    @override
    def copy(self, child: ExpressionTree):
        return init_expression(child=child)

    def parametrize(self, variable: Symbol | str) -> VariableExpression:
        if not isinstance(variable, Symbol):
            variable = Symbol(variable)

        expr = super().parametrize(variable)  # type: ignore

        return init_variable_expression(
            child=expr.child,
            symbol=variable,
        )


def init_expression(child: ExpressionTree):
    return ExpressionImpl(
        child=child,
    )


@dataclassabc(frozen=True)
class VariableExpressionImpl(VariableExpression):
    child: ExpressionTree
    symbol: Symbol

    @override
    def copy(self, child: ExpressionTree):
        return init_expression(child=child)


def init_variable_expression(child: ExpressionTree, symbol: Symbol):
    return VariableExpressionImpl(
        child=child,
        symbol=symbol,
    )
