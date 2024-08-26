from abc import ABC, abstractmethod
from typing import override

from polymat.sparserepr.data.polynomial import MaybePolynomialType
from polymat.sparserepr.data.polynomialmatrix import PolynomialMatrixType
from polymat.sparserepr.sparserepr import SparseRepr


class FromPolynomialMatrixMixin(SparseRepr, ABC):
    """Matrix with polynomial entries, stored as a dictionary."""

    @property
    @abstractmethod
    def data(self) -> PolynomialMatrixType:
        """Get the dictionary."""

    @override
    def at(self, row: int, col: int) -> MaybePolynomialType:
        """See :py:meth:`PolyMatrixMixin.at`."""

        return self.data.get((row, col))
