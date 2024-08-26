from abc import abstractmethod
from typing import override

from polymat.sparserepr.data.polynomial import MaybePolynomialType
from polymat.sparserepr.sparserepr import SingleChildSparseReprMixin


class RepMatSparseReprMixin(SingleChildSparseReprMixin):
    @property
    @abstractmethod
    def child_shape(self) -> tuple[int, int]: ...

    @override
    def at(self, row: int, col: int) -> MaybePolynomialType:
        n_row, n_col = self.child_shape

        rel_row = row % n_row
        rel_col = col % n_col

        return self.child.at(row=rel_row, col=rel_col)
