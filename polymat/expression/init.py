from typing import override
from dataclassabc import dataclassabc

from polymat.expression.abc import (
    MatrixExpression,
    VariableExpression,
)
from polymat.expressiontree.nodes import ExpressionNode
from polymat.symbol import Symbol


@dataclassabc(frozen=True)
class ExpressionImpl(MatrixExpression):
    child: ExpressionNode

    @override
    def copy(self, child: ExpressionNode):
        return init_expression(child=child)

    def parametrize(self, variable: Symbol | str) -> VariableExpression:
        if not isinstance(variable, Symbol):
            variable = Symbol(variable)

        expr = super().parametrize(variable)  # type: ignore

        return init_variable_expression(
            child=expr.child,
            symbol=variable,
        )


def init_expression(child: ExpressionNode):
    return ExpressionImpl(
        child=child,
    )


@dataclassabc(frozen=True)
class VariableExpressionImpl(VariableExpression):
    child: ExpressionNode
    symbol: Symbol

    @override
    def copy(self, child: ExpressionNode):
        return init_expression(child=child)


def init_variable_expression(child: ExpressionNode, symbol: Symbol):
    return VariableExpressionImpl(
        child=child,
        symbol=symbol,
    )
