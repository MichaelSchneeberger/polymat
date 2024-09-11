from abc import abstractmethod
from typing import override

from polymat.expressiontree.operations.filtermixin import FilterMixin
from polymat.sparserepr.data.polynomial import PolynomialType
from polymat.utils.getstacklines import to_operator_traceback


class FilterPredicate(FilterMixin):
    PredicatorType = tuple[bool | int, ...]

    @property
    @abstractmethod
    def predicate(self) -> PredicatorType: ...

    def __str__(self):
        return f"filter_predicate({self.child}, {self.predicate})"

    @override
    def _assert_nrows(self, n_rows: int) -> None:
        if not (n_rows == len(self.predicate)):
            raise AssertionError(
                to_operator_traceback(
                    message=f"{n_rows=} is not {len(self.predicate)=}",
                    stack=self.stack,
                )
            )

    @override
    def _filter(self, row: int, polynomial: PolynomialType) -> bool:
        return bool(self.predicate[row])
