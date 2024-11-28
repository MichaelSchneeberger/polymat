from typing import Iterable
from dataclassabc import dataclassabc

from polymat.sparserepr.data.polynomialmatrix import (
    MatrixIndexType,
    PolynomialMatrixType,
    polynomial_matrix_from_iterable,
)
from polymat.sparserepr.data.polynomial import PolynomialType
from polymat.sparserepr.operations.blockdiagonalsparsereprmixin import (
    BlockDiagonalSparseReprMixin,
)
from polymat.sparserepr.operations.broadcastsparsereprmixin import (
    BroadcastSparseReprMixin,
)
from polymat.sparserepr.operations.diagmatrixfromvecsparsereprmixin import (
    DiagMatrixFromVecSparseReprMixin,
)
from polymat.sparserepr.operations.kronsparsereprmixin import KronSparseReprMixin
from polymat.sparserepr.operations.repmatsparsereprmixin import (
    RepMatSparseReprMixin,
)
from polymat.sparserepr.operations.reshapesparsereprmixin import (
    ReshapeSparseReprMixin,
)
from polymat.sparserepr.operations.getitemsparsereprmixin import (
    GetItemSparseReprMixin,
)
from polymat.sparserepr.operations.symmetricsparsereprmixin import (
    SymmetricSparseReprMixin,
)
from polymat.sparserepr.operations.transposesparsereprmixin import (
    TransposeSparseReprMixin,
)
from polymat.sparserepr.operations.vecfromdiagmatrixsparsereprmixin import (
    VecFromDiagMatrixSparseReprMixin,
)
from polymat.sparserepr.operations.frompolynomialmixin import (
    FromPolynomialMatrixMixin,
)
from polymat.sparserepr.sparserepr import SparseRepr
from polymat.sparserepr.operations.vstacksparsereprmixin import (
    VStackSparseReprMixin,
)


@dataclassabc(frozen=True, slots=True)
class BlockDiagonalSparseReprImpl(BlockDiagonalSparseReprMixin):
    children: tuple[SparseRepr]
    row_col_ranges: tuple[tuple[range, range], ...]
    shape: tuple[int, int]


def init_block_diagonal_sparse_repr(
    children: tuple[SparseRepr],
    row_col_ranges: tuple[tuple[range, range], ...],
    shape: tuple[int, int],
):
    return BlockDiagonalSparseReprImpl(children=children, row_col_ranges=row_col_ranges, shape=shape)


@dataclassabc(frozen=True, slots=True)
class BroadcastSparseReprImpl(BroadcastSparseReprMixin):
    polynomial: PolynomialType
    shape: tuple[int, int]


def init_broadcast_sparse_repr(
    polynomial: PolynomialType,
    shape: tuple[int, int],
):
    return BroadcastSparseReprImpl(polynomial=polynomial, shape=shape)


@dataclassabc(frozen=True, slots=True)
class DiagMatrixFromVecSparseReprImpl(DiagMatrixFromVecSparseReprMixin):
    child: SparseRepr
    shape: tuple[int, int]


def init_diag_matrix_from_vec_sparse_repr(
    child: SparseRepr,
    shape: tuple[int, int],
):
    return DiagMatrixFromVecSparseReprImpl(child=child, shape=shape)


@dataclassabc(frozen=True, slots=True)
class KronSparseReprImpl(KronSparseReprMixin):
    left: SparseRepr
    right: SparseRepr
    shape: tuple[int, int]


def init_kron_sparse_repr(
    left: SparseRepr,
    right: SparseRepr,
    shape: tuple[int, int],
):
    return KronSparseReprImpl(left=left, right=right, shape=shape)


@dataclassabc(frozen=True, slots=True)
class GetItemSparseReprImpl(GetItemSparseReprMixin):
    child: SparseRepr
    key: tuple[tuple[int, ...], tuple[int, ...]]
    shape: tuple[int, int]


def init_get_item_sparse_repr(
    child: SparseRepr,
    key: tuple[tuple[int, ...], tuple[int, ...]],
    shape: tuple[int, int],
):
    return GetItemSparseReprImpl(child=child, key=key, shape=shape)


@dataclassabc(frozen=True, slots=True)
class FromPolynomialMatrixImpl(FromPolynomialMatrixMixin):
    data: PolynomialMatrixType
    shape: tuple[int, int]


def init_sparse_repr_from_data(
    data: PolynomialMatrixType,
    shape: tuple[int, int],
):
    return FromPolynomialMatrixImpl(data=data, shape=shape)


def init_sparse_repr_from_iterable(
    data: Iterable[tuple[MatrixIndexType, PolynomialType]],
    shape: tuple[int, int],
):
    result = polynomial_matrix_from_iterable(data)

    return FromPolynomialMatrixImpl(data=result, shape=shape)


@dataclassabc(frozen=True, slots=True)
class SymmetricSparseReprImpl(SymmetricSparseReprMixin):
    child: SparseRepr


def init_symmetric_sparse_repr(child: SparseRepr):
    return SymmetricSparseReprImpl(child=child)


@dataclassabc(frozen=True, slots=True)
class RepMatSparseReprImpl(RepMatSparseReprMixin):
    child: SparseRepr
    child_shape: tuple[int, int]
    shape: tuple[int, int]


def init_repmat_sparse_repr(
    child: SparseRepr,
    child_shape: tuple[int, int],
    shape: tuple[int, int],
):
    return RepMatSparseReprImpl(child=child, child_shape=child_shape, shape=shape)


@dataclassabc(frozen=True, slots=True)
class ReshapeSparseReprImpl(ReshapeSparseReprMixin):
    child: SparseRepr
    shape: tuple[int, int]


def init_reshape_sparse_repr(
    child: SparseRepr,
    shape: tuple[int, int],
):
    return ReshapeSparseReprImpl(
        child=child,
        shape=shape,
    )


@dataclassabc(frozen=True, slots=True)
class TransposeSparseReprImpl(TransposeSparseReprMixin):
    child: SparseRepr


def init_transpose_sparse_repr(child: SparseRepr):
    return TransposeSparseReprImpl(child=child)


@dataclassabc(frozen=True, slots=True)
class VecFromDiagMatrixSparseReprImpl(VecFromDiagMatrixSparseReprMixin):
    child: SparseRepr
    shape: tuple[int, int]


def init_vec_from_diag_matrix_sparse_repr(
    child: SparseRepr,
    shape: tuple[int, int],
):
    return VecFromDiagMatrixSparseReprImpl(child=child, shape=shape)


@dataclassabc(frozen=True, slots=True)
class VStackSparseReprImpl(VStackSparseReprMixin):
    children: tuple[SparseRepr, ...]
    row_ranges: tuple[range, ...]
    shape: tuple[int, int]


def init_vstack_sparse_repr(
    children: tuple[SparseRepr, ...],
    row_ranges: tuple[range, ...],
    shape: tuple[int, int],
):
    return VStackSparseReprImpl(children=children, row_ranges=row_ranges, shape=shape)
