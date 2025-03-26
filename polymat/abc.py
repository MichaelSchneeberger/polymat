"""
This module (`abc.py`) contains abstract base classes (ABCs) that are designed to be extended
through inheritance in other Python projects.
"""

from utils.getstacklines import FrameSummaryMixin as _FrameSummaryMixin
from polymat.expressiontree.nodes import (
    ExpressionNode as _ExpressionNode,
    SingleChildExpressionNode as _SingleChildExpressionNode,
)
from polymat.expression.expression import (
    Expression as _Expression,
    VariableExpression as _VariableExpression,
)


FrameSummaryMixin = _FrameSummaryMixin

ExpressionNode = _ExpressionNode
SingleChildExpressionNode = _SingleChildExpressionNode

Expression = _Expression
VariableExpression = _VariableExpression
