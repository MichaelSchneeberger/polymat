from typing_extensions import override

from polymat.sparserepr.data.polynomial import MaybePolynomialType
from polymat.sparserepr.sparsereprmixin import SingleChildSparseReprMixin


class DiagMatrixFromVecSparseReprMixin(SingleChildSparseReprMixin):
    @override
    def at(self, row: int, col: int) -> MaybePolynomialType:
        if row == col:
            return self.child.at(row, 0)
