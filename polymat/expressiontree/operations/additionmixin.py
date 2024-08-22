from polymat.expressiontree.operations.elementwiseop import ElementwiseOpMixin
from polymat.sparserepr.data.polynomial import (
    MaybePolynomialType,
    add_maybe_polynomials,
)


class AdditionMixin(ElementwiseOpMixin):
    @staticmethod
    def operator(
        left: MaybePolynomialType, right: MaybePolynomialType
    ) -> MaybePolynomialType:
        return add_maybe_polynomials(left, right)

    @property
    def operator_name(self) -> str:
        return "add"

    @property
    def is_addition(self) -> bool:
        return True
