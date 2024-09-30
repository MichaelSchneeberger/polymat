import math
from typing import override

from polymat.sparserepr.data.polynomial import PolynomialType, is_zero
from polymat.expressiontree.operations.filtermixin import FilterMixin


class FilterNonZero(FilterMixin):
    def __str__(self):
        return f"filter_non_zero({self.child})"

    @override
    def _assert_nrows(self, n_rows: int) -> None: ...

    @override
    def _filter(self, row: int, polynomial: PolynomialType) -> bool:
        return not is_zero(polynomial)
