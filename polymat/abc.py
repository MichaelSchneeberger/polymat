"""
This module (`abc.py`) contains abstract base classes (ABCs) that are designed to be extended
through inheritance in other Python projects.
"""

from polymat.symbols.symbol import Symbol as _Symbol
from polymat.utils.getstacklines import FrameSummaryMixin as _FrameSummaryMixin
from polymat.expressiontree.nodes import (
    ExpressionNode as _ExpressionNode,
    SingleChildExpressionNode as _SingleChildExpressionNode,
)
from polymat.expression.expression import (
    Expression as _Expression,
    VariableExpression as _VariableExpression,
)

Symbol = _Symbol

FrameSummaryMixin = _FrameSummaryMixin

ExpressionNode = _ExpressionNode
SingleChildExpressionNode = _SingleChildExpressionNode

Expression = _Expression
VariableExpression = _VariableExpression
