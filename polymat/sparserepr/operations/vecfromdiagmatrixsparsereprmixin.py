from typing_extensions import override

from polymat.sparserepr.data.polynomial import MaybePolynomialType
from polymat.sparserepr.sparsereprmixin import SingleChildSparseReprMixin


class VecFromDiagMatrixSparseReprMixin(SingleChildSparseReprMixin):
    @override
    def at(self, row: int, col: int) -> MaybePolynomialType:
        return self.child.at(row, row)
