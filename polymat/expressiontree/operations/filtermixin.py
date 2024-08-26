from abc import abstractmethod
from typing import override

from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.expressiontree.expressiontree import SingleChildExpressionTreeMixin
from polymat.utils.getstacklines import FrameSummaryMixin, to_operator_traceback
from polymat.sparserepr.init import init_from_polynomial_matrix


class FilterMixin(FrameSummaryMixin, SingleChildExpressionTreeMixin):
    PREDICATOR_TYPE = tuple[bool | int, ...]

    @property
    @abstractmethod
    def predicator(self) -> PREDICATOR_TYPE: ...

    def __str__(self):
        return f"filter({self.child})"

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, child = self.child.apply(state=state)

        if not (child.shape[1] == 1):
            raise AssertionError(
                to_operator_traceback(
                    message=f"{child.shape[1]=} is not 1",
                    stack=self.stack,
                )
            )

        if not (child.shape[0] == len(self.predicator)):
            raise AssertionError(
                to_operator_traceback(
                    message=f"{child.shape[0]=} is not {len(self.predicator)=}",
                    stack=self.stack,
                )
            )

        def gen_polynomial_matrix():
            row = 0
            for (index_row, _), polynomial in child.entries():
                if self.predicator[index_row]:
                    yield (row, 0), polynomial
                    row += 1

        polymatrix = dict(gen_polynomial_matrix())

        n_row = len(polymatrix)
        n_col = 0 if n_row == 0 else 1

        return state, init_from_polynomial_matrix(
            data=polymatrix,
            shape=(n_row, n_col),
        )
