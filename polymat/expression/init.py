from typing import override
from dataclassabc import dataclassabc

from polymat.expression.abc import (
    Expression,
    VariableExpression,
)
from polymat.expressiontree.expressiontreemixin import ExpressionTreeMixin
from polymat.variable import Variable


@dataclassabc(frozen=True)
class ExpressionImpl(Expression):
    child: ExpressionTreeMixin

    @override
    def copy(self, child: ExpressionTreeMixin):
        return init_expression(child=child)

    def parametrize(self, variable: Variable | str) -> VariableExpression:
        if not isinstance(variable, Variable):
            variable = Variable(variable)

        expr = super().parametrize(variable)  # type: ignore

        return init_variable_expression(
            child=expr.child,
            variable=variable,
        )


def init_expression(child: ExpressionTreeMixin):
    return ExpressionImpl(
        child=child,
    )


@dataclassabc(frozen=True)
class VariableExpressionImpl(VariableExpression):
    child: ExpressionTreeMixin
    variable: Variable

    @override
    def copy(self, child: ExpressionTreeMixin):
        return init_expression(child=child)


def init_variable_expression(child: ExpressionTreeMixin, variable: Variable):
    return VariableExpressionImpl(
        child=child,
        variable=variable,
    )
