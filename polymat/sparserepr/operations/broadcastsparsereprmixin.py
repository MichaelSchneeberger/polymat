from abc import ABC, abstractmethod
from typing import override

from polymat.sparserepr.data.polynomial import MaybePolynomialType, PolynomialType
from polymat.sparserepr.sparserepr import SparseRepr


class BroadcastSparseReprMixin(SparseRepr, ABC):
    """
    TODO: docstring, similar to numpy broadcasting.
    https://numpy.org/doc/stable/user/basics.broadcasting.html
    """

    @property
    @abstractmethod
    def polynomial(self) -> PolynomialType:
        """Scalar polynomial that is broadcasted."""

    @override
    def at(self, row: int, col: int) -> MaybePolynomialType:
        """See :py:meth:`PolyMatrixMixin.at`."""

        # copy polynomial
        return self.polynomial
