"""
The following assignments serve as placeholders for classes that are defined in the stub file `typedexpressions.pyi`. 
These placeholders are used only for type hinting and should have no impact on the runtime behavior.
They are dedeclared here to enable imports without requiring a check for the typing.TYPE_CHECKING constant.
"""

from polymat.expression.expression import (
    Expression as _Expression,
    VariableExpression as _VariableExpression,
)


MatrixExpression = _Expression
SymmetricMatrixExpression = MatrixExpression
VectorExpression = MatrixExpression
RowVectorExpression = MatrixExpression
ScalarPolynomialExpression = MatrixExpression
MonomialVectorExpression = MatrixExpression
MonomialExpression = MatrixExpression
VariableVectorExpression = MatrixExpression
VariableMatrixExpression = MatrixExpression
VariableVectorSymbolExpression = _VariableExpression
VariableExpression = VariableVectorSymbolExpression
