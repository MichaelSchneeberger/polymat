"""
This module (`typing.py`) provides support for type hints in Python, which are used to
specify the expected types of variables, function arguments, and return values.
"""

from polymat.symbol import Symbol as _Symbol
from polymat.state.state import State as _State
from polymat.arrayrepr.arrayrepr import ArrayRepr as _ArrayRepr
from polymat.sparserepr.sparserepr import SparseRepr as _SparseRepr
from polymat.expressiontree.nodes import (
    ExpressionNode as _ExpressionNode,
)
from polymat.expression.typedexpressions import (
    MatrixExpression as _MatrixExpression,
    SymmetricMatrixExpression as _SymmetricMatrixExpression,
    VectorExpression as _VectorExpression,
    RowVectorExpression as _RowVectorExpression,
    ScalarPolynomialExpression as _ScalarPolynomialExpression,
    VariableVectorExpression as _VariableVectorExpression,
    VariableExpression as _VariableExpression,
    VariableVectorSymbolExpression as _VariableVectorSymbolExpression,
    MonomialVectorExpression as _MonomialVectorExpression,
    MonomialExpression as _MonomialExpression,
)

State = _State
Symbol = _Symbol

ArrayRepr = _ArrayRepr

SparseRepr = _SparseRepr

ExpressionNode = _ExpressionNode

MatrixExpression = _MatrixExpression
SymmetricMatrixExpression = _SymmetricMatrixExpression
VectorExpression = _VectorExpression
RowVectorExpression = _RowVectorExpression
ScalarPolynomialExpression = _ScalarPolynomialExpression
VariableVectorExpression = _VariableVectorExpression
MonomialVectorExpression = _MonomialVectorExpression
VariableExpression = _VariableExpression
VariableVectorSymbolExpression = _VariableVectorSymbolExpression
MonomialExpression = _MonomialExpression
