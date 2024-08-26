from typing import override
from dataclassabc import dataclassabc

from polymat.expression.abc import (
    Expression,
    VariableExpression,
)
from polymat.expressiontree.expressiontreemixin import ExpressionTreeMixin
from polymat.symbol import Symbol


@dataclassabc(frozen=True)
class ExpressionImpl(Expression):
    child: ExpressionTreeMixin

    @override
    def copy(self, child: ExpressionTreeMixin):
        return init_expression(child=child)

    def parametrize(self, variable: Symbol | str) -> VariableExpression:
        if not isinstance(variable, Symbol):
            variable = Symbol(variable)

        expr = super().parametrize(variable)  # type: ignore

        return init_variable_expression(
            child=expr.child,
            symbol=variable,
        )


def init_expression(child: ExpressionTreeMixin):
    return ExpressionImpl(
        child=child,
    )


@dataclassabc(frozen=True)
class VariableExpressionImpl(VariableExpression):
    child: ExpressionTreeMixin
    symbol: Symbol

    @override
    def copy(self, child: ExpressionTreeMixin):
        return init_expression(child=child)


def init_variable_expression(child: ExpressionTreeMixin, symbol: Symbol):
    return VariableExpressionImpl(
        child=child,
        symbol=symbol,
    )
