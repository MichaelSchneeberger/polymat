from abc import abstractmethod
from typing import override

from polymat.sparserepr.data.polynomial import MaybePolynomialType
from polymat.sparserepr.sparserepr import SingleChildSparseReprMixin


class GetItemSparseReprMixin(SingleChildSparseReprMixin):
    @property
    @abstractmethod
    def key(self) -> tuple[tuple[int, ...], tuple[int, ...]]: ...

    @override
    def at(self, row: int, col: int) -> MaybePolynomialType:
        ref_row = self.key[0][row]
        ref_col = self.key[1][col]

        return self.child.at(row=ref_row, col=ref_col)
