from abc import abstractmethod
from typing import override

from polymat.sparserepr.data.polynomial import MaybePolynomialType
from polymat.sparserepr.sparsereprmixin import MultiChildrenSparseReprMixin


class BlockDiagonalSparseReprMixin(MultiChildrenSparseReprMixin):
    @property
    @abstractmethod
    def row_col_ranges(self) -> tuple[tuple[range, range], ...]: ...

    @override
    def at(self, row: int, col: int) -> MaybePolynomialType:
        for (row_range, col_range), pm in zip(self.row_col_ranges, self.children):
            if row in row_range and col in col_range:
                block_row = row - row_range.start
                block_col = col - col_range.start
                return pm.at(block_row, block_col)
