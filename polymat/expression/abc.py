from abc import abstractmethod

from polymat.expression.expression import Expression
from polymat.variable import Variable


class SymmetricExpression(Expression):
    pass


class VectorExpression(Expression):
    pass


class RowVectorExpression(Expression):
    pass


class PolynomialExpression(VectorExpression):
    pass


class MonomialVectorExpression(VectorExpression):
    pass


class MonomialExpression(PolynomialExpression, MonomialVectorExpression):
    pass


class VariableVectorExpression(MonomialVectorExpression):
    pass


class VariableExpression(VariableVectorExpression):
    @property
    @abstractmethod
    def variable(self) -> Variable: ...


class SingleDimVariableExpression(VariableExpression):
    pass
