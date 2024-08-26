from typing import override

from polymat.sparserepr.data.polynomial import MaybePolynomialType, multiply_polynomials
from polymat.sparserepr.sparserepr import TwoChildrenSparseReprMixin


class KronSparseReprMixin(TwoChildrenSparseReprMixin):
    @override
    def at(self, row: int, col: int) -> MaybePolynomialType:
        left_row = row // self.right.shape[0]
        right_row = row - left_row * self.right.shape[0]

        left_col = col // self.right.shape[1]
        right_col = col - left_col * self.right.shape[1]

        left = self.left.at(left_row, left_col)
        right = self.right.at(right_row, right_col)

        if left and right:
            return multiply_polynomials(left, right)
