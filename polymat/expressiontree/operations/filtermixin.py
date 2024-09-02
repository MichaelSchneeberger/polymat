from abc import abstractmethod
from typing import override

from polymat.sparserepr.data.polynomial import PolynomialType
from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.expressiontree.nodes import SingleChildExpressionNode
from polymat.utils.getstacklines import FrameSummaryMixin, to_operator_traceback
from polymat.sparserepr.init import init_from_polynomial_matrix


class FilterMixin(FrameSummaryMixin, SingleChildExpressionNode):
    @abstractmethod
    def _assert_nrows(self, n_rows: int) -> None: ...

    @abstractmethod
    def _filter(self, row: int, polynomial: PolynomialType) -> bool: ...

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, child = self.child.apply(state=state)

        n_rows, n_cols = child.shape

        if not (child.shape[1] == 1):
            raise AssertionError(
                to_operator_traceback(
                    message=f"{n_cols=} is not 1",
                    stack=self.stack,
                )
            )

        self._assert_nrows(n_rows)

        def gen_polynomial_matrix():
            row = 0
            for (index_row, _), polynomial in child.entries():
                if self._filter(index_row, polynomial):
                    yield (row, 0), polynomial
                    row += 1

        polymatrix = dict(gen_polynomial_matrix())

        n_row = len(polymatrix)
        n_col = 0 if n_row == 0 else 1

        return state, init_from_polynomial_matrix(
            data=polymatrix,
            shape=(n_row, n_col),
        )
