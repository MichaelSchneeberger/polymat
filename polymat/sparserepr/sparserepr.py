from abc import ABC, abstractmethod
from typing import Iterable

from polymat.sparserepr.data.monomial import MonomialType
from polymat.sparserepr.data.polynomialmatrix import MatrixIndexType
from polymat.sparserepr.data.polynomial import MaybePolynomialType, PolynomialType


class SparseRepr(ABC):
    """
    Matrix with polynomial entries.
    """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]: ...

    def __iter__(self):
        return self.entries()

    @abstractmethod
    def at(self, row: int, col: int) -> MaybePolynomialType:
        """Return the polynomial at the entry (row, col).
        If the entry is zero it returns an empty `PolyDict`, i.e. an empty
        dictionary `{}`.

        .. code:: py

            p.at(row, col) # if you have row and col
            p.at(*entry) # if you have a MatrixIndex
        """

    def entries(self) -> Iterable[tuple[MatrixIndexType, PolynomialType]]:
        """
        Iterate over the non-zero entries of the polynomial matrix.

        Typical usage looks like this (`p` is a polymatrix):

        .. code:: py

            for entry, poly in p.entries():
                for monomial, coeff in poly.terms():
                    # do something with coefficients
        """

        nrows, ncols = self.shape
        for row in range(nrows):
            for col in range(ncols):
                polynomial = self.at(row, col)

                if polynomial:
                    yield (row, col), polynomial

    @property
    def n_entries(self) -> int:
        nrows, ncols = self.shape
        return nrows * ncols
    
    def to_monomials(self) -> Iterable[MonomialType]:
        for _, polynomial in self.entries():
            yield from polynomial.keys()

    def to_indices(self) -> Iterable[int]:
        for monomial in self.to_monomials():
            if len(monomial) == 0:
                continue

            for index, _ in monomial:
                yield index


class TwoChildrenSparseReprMixin(SparseRepr):
    @property
    @abstractmethod
    def left(self) -> SparseRepr: ...

    @property
    @abstractmethod
    def right(self) -> SparseRepr: ...


class SingleChildSparseReprMixin(SparseRepr):
    @property
    @abstractmethod
    def child(self) -> SparseRepr: ...


class MultiChildrenSparseReprMixin(SparseRepr):
    @property
    @abstractmethod
    def children(self) -> tuple[SparseRepr, ...]: ...
