from abc import abstractmethod

from polymat.expression.expression import Expression as _Expression
from polymat.variable import Variable


class VariableExpression(_Expression):
    @property
    @abstractmethod
    def variable(self) -> Variable: ...


# These global variables serve as placeholders for classes that are defined only in the stub file for type checking. 
# They are irrelevant during code execution but are declared here to allow imports without checking the typing.TYPE_CHECKING constant.
Expression = _Expression
SymmetricExpression = _Expression
VectorExpression = _Expression
RowVectorExpression = _Expression
PolynomialExpression = _Expression
MonomialVectorExpression = _Expression
MonomialExpression = _Expression
VariableVectorExpression = _Expression
SingleDimVariableExpression = VariableExpression
