from polymat.expressiontree.operations.elementwiseopmixin import ElementwiseOpMixin
from polymat.sparserepr.data.polynomial import MaybePolynomialType, multiply_polynomials


class ElementwiseMult(ElementwiseOpMixin):
    @staticmethod
    def operator(
        left: MaybePolynomialType, right: MaybePolynomialType
    ) -> MaybePolynomialType:
        if left and right:
            return multiply_polynomials(left, right)

    @property
    def operator_name(self) -> str:
        return "mul"

    @property
    def is_addition(self) -> bool:
        return False
