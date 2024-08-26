from typing import override

from polymat.sparserepr.data.polynomial import (
    MaybePolynomialType,
    add_polynomials,
    multiply_with_scalar,
    multiply_with_scalar_mutable,
)
from polymat.sparserepr.sparserepr import SingleChildSparseReprMixin


class SymmetricSparseReprMixin(SingleChildSparseReprMixin):
    @property
    def shape(self) -> tuple[int, int]:
        max_dim = max(self.child.shape)
        return max_dim, max_dim

    @override
    def at(self, row: int, col: int) -> MaybePolynomialType:
        left = self.child.at(row, col)
        right = self.child.at(col, row)

        scalar = 0.5

        match (left, right):
            case (dict() as polynomial, None) | (None, dict() as polynomial):
                return multiply_with_scalar(polynomial, scalar)

            case (dict(), dict()):
                mutable = add_polynomials(left, right)

                if mutable:
                    return multiply_with_scalar_mutable(mutable=mutable, scalar=0.5)
