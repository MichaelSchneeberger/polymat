from abc import abstractmethod
from typing import override

from polymat.sparserepr.data.polynomial import MaybePolynomialType
from polymat.sparserepr.sparsereprmixin import SingleChildSparseReprMixin


class ReshapeSparseReprMixin(SingleChildSparseReprMixin):
    @property
    @abstractmethod
    def child_shape(self) -> tuple[int, int]: ...

    @override
    def at(self, row: int, col: int) -> MaybePolynomialType:
        index = row + self.shape[0] * col

        child_col = int(index / self.child_shape[0])
        child_row = index - child_col * self.child_shape[0]

        return self.child.at(row=child_row, col=child_col)
