from typing import override
from dataclassabc import dataclassabc

from polymat.expression.expression import Expression
from polymat.expression.typedexpressions import (
    VariableExpression,
)
from polymat.expressiontree.nodes import ExpressionNode
from polymat.symbol import Symbol


@dataclassabc(frozen=True)
class ExpressionImpl(Expression):
    child: ExpressionNode

    @override
    def copy(self, child: ExpressionNode):
        return init_expression(child=child)


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
