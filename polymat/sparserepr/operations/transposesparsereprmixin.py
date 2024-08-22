from functools import cached_property

from typing_extensions import override

from polymat.sparserepr.data.polynomial import MaybePolynomialType
from polymat.sparserepr.sparsereprmixin import SingleChildSparseReprMixin


class TransposeSparseReprMixin(SingleChildSparseReprMixin):
    @cached_property
    def shape(self) -> tuple[int, int]:
        return self.child.shape[1], self.child.shape[0]

    @override
    def at(self, row: int, col: int) -> MaybePolynomialType:
        return self.child.at(row=col, col=row)
