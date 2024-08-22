from typing import override

from polymat.sparserepr.data.polynomial import (
    MaybePolynomialType,
    add_polynomials,
    multiply_with_scalar_mutable,
)
from polymat.sparserepr.sparsereprmixin import SingleChildSparseReprMixin


class SymmetricSparseReprMixin(SingleChildSparseReprMixin):
    @property
    def shape(self) -> tuple[int, int]:
        max_dim = max(self.child.shape)
        return max_dim, max_dim

    @override
    def at(self, row: int, col: int) -> MaybePolynomialType:
        left = self.child.at(row, col)
        right = self.child.at(col, row)

        if left and right:
            result = add_polynomials(left, right)

        elif left:
            result = left

        elif right:
            result = right

        else:
            result = None

        if result:
            return multiply_with_scalar_mutable(result, 0.5)
