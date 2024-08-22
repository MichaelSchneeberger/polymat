from abc import abstractmethod
from typing_extensions import override

from polymat.sparserepr.data.polynomial import MaybePolynomialType
from polymat.sparserepr.sparsereprmixin import MultiChildrenSparseReprMixin


class VStackSparseReprMixin(MultiChildrenSparseReprMixin):
    @property
    @abstractmethod
    def row_ranges(self) -> tuple[range, ...]: ...

    @override
    def at(self, row: int, col: int) -> MaybePolynomialType:
        for polymatrix, block_range in zip(self.children, self.row_ranges):
            if block_range.start <= row < block_range.stop:
                return polymatrix.at(
                    row=row - block_range.start,
                    col=col,
                )
