from typing import override

from polymat.sparserepr.data.polynomial import MaybePolynomialType
from polymat.sparserepr.sparserepr import SingleChildSparseReprMixin


class ReshapeSparseReprMixin(SingleChildSparseReprMixin):
    @override
    def at(self, row: int, col: int) -> MaybePolynomialType:
        index = row + self.shape[0] * col

        child_col = int(index / self.child.shape[0])
        child_row = index - child_col * self.child.shape[0]

        return self.child.at(row=child_row, col=child_col)
