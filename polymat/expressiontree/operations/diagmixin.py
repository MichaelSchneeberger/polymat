from typing import override

from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.expressiontree.expressiontree import SingleChildExpressionTreeMixin
from polymat.sparserepr.init import (
    init_diag_matrix_from_vec_sparse_repr,
    init_vec_from_diag_matrix_sparse_repr,
)
from polymat.utils.getstacklines import FrameSummaryMixin, to_operator_traceback


class DiagMixin(FrameSummaryMixin, SingleChildExpressionTreeMixin):
    """
    [[1],[2]]  ->  [[1,0],[0,2]]

    or

    [[1,0],[0,2]]  ->  [[1],[2]]
    """

    def __str__(self):
        return f"diag({self.child})"

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, child = self.child.apply(state=state)

        # Vector to diagonal matrix
        if child.shape[1] == 1:
            return state, init_diag_matrix_from_vec_sparse_repr(
                child=child, shape=(child.shape[0], child.shape[0])
            )

        # Diagonal matrix to vector
        else:
            if not (child.shape[0] == child.shape[1]):
                raise AssertionError(
                    to_operator_traceback(
                        message=f"{child.shape[0]=} is not {child.shape[1]=}",
                        stack=self.stack,
                    )
                )

            return state, init_vec_from_diag_matrix_sparse_repr(
                child=child, shape=(child.shape[0], 1)
            )
