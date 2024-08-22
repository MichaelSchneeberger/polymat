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
from polymat.sparserepr.operations.sparsereprfrompolynomialmixin import (
    SparseReprFromPolynomialMatrixMixin,
)
from polymat.sparserepr.sparsereprmixin import SparseReprMixin
from polymat.sparserepr.operations.vstacksparsereprmixin import (
    VStackSparseReprMixin,
)
from typing import Iterable


@dataclassabc(frozen=True)
class BlockDiagonalSparseReprImpl(BlockDiagonalSparseReprMixin):
    children: tuple[SparseReprMixin]
    row_col_ranges: tuple[tuple[range, range], ...]
    shape: tuple[int, int]


init_block_diagonal_sparse_repr = BlockDiagonalSparseReprImpl


@dataclassabc(frozen=True)
class BroadcastSparseReprImpl(BroadcastSparseReprMixin):
    polynomial: PolynomialType
    shape: tuple[int, int]


init_broadcast_sparse_repr = BroadcastSparseReprImpl


@dataclassabc(frozen=True)
class DiagMatrixFromVecSparseReprImpl(DiagMatrixFromVecSparseReprMixin):
    child: SparseReprMixin
    shape: tuple[int, int]


init_diag_matrix_from_vec_sparse_repr = DiagMatrixFromVecSparseReprImpl


@dataclassabc(frozen=True)
class KronSparseReprImpl(KronSparseReprMixin):
    left: SparseReprMixin
    right: SparseReprMixin
    shape: tuple[int, int]


init_kron_sparse_repr = KronSparseReprImpl


@dataclassabc(frozen=True)
class GetItemSparseReprImpl(GetItemSparseReprMixin):
    child: SparseReprMixin
    key: tuple[tuple[int, ...], tuple[int, ...]]
    shape: tuple[int, int]


init_get_item_sparse_repr = GetItemSparseReprImpl


@dataclassabc(frozen=True)
class SparseReprFromPolynomialMatrixImpl(SparseReprFromPolynomialMatrixMixin):
    data: PolynomialMatrixType
    shape: tuple[int, int]


def init_sparse_repr_from_data(
    data: PolynomialMatrixType,
    shape: tuple[int, int],
):
    return SparseReprFromPolynomialMatrixImpl(data=data, shape=shape)


def init_sparse_repr_from_iterable(
    data: Iterable[tuple[MatrixIndexType, PolynomialType]],
    shape: tuple[int, int],
):
    result = polynomial_matrix_from_iterable(data)

    return SparseReprFromPolynomialMatrixImpl(data=result, shape=shape)


@dataclassabc(frozen=True)
class SymmetricSparseReprImpl(SymmetricSparseReprMixin):
    child: SparseReprMixin


init_symmetric_sparse_repr = SymmetricSparseReprImpl


@dataclassabc(frozen=True)
class RepMatSparseReprImpl(RepMatSparseReprMixin):
    child: SparseReprMixin
    child_shape: tuple[int, int]
    shape: tuple[int, int]


init_repmat_sparse_repr = RepMatSparseReprImpl


@dataclassabc(frozen=True)
class ReshapeSparseReprImpl(ReshapeSparseReprMixin):
    child: SparseReprMixin
    child_shape: tuple[int, int]
    shape: tuple[int, int]


init_reshape_sparse_repr = ReshapeSparseReprImpl


@dataclassabc(frozen=True)
class TransposeSparseReprImpl(TransposeSparseReprMixin):
    child: SparseReprMixin


init_transpose_sparse_repr = TransposeSparseReprImpl


@dataclassabc(frozen=True)
class VecFromDiagMatrixSparseReprImpl(VecFromDiagMatrixSparseReprMixin):
    child: SparseReprMixin
    shape: tuple[int, int]


init_vec_from_diag_matrix_sparse_repr = VecFromDiagMatrixSparseReprImpl


@dataclassabc(frozen=True)
class VStackSparseReprImpl(VStackSparseReprMixin):
    children: tuple[SparseReprMixin, ...]
    row_ranges: tuple[range, ...]
    shape: tuple[int, int]


init_vstack_sparse_repr = VStackSparseReprImpl
