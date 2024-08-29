from typing import override

from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.expressiontree.nodes import SingleChildExpressionNode
from polymat.sparserepr.init import (
    init_diag_matrix_from_vec_sparse_repr,
    init_transpose_sparse_repr,
    init_vec_from_diag_matrix_sparse_repr,
)
from polymat.utils.getstacklines import FrameSummaryMixin, to_operator_traceback


class DiagMixin(FrameSummaryMixin, SingleChildExpressionNode):
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

        match child.shape:

            # Vector to diagonal matrix
            case (n_row, 1):
                return state, init_diag_matrix_from_vec_sparse_repr(
                    child=child, shape=(n_row, n_row)
                )
            case (1, n_col):
                return state, init_diag_matrix_from_vec_sparse_repr(
                    child=init_transpose_sparse_repr(child), 
                    shape=(n_col, n_col)
                )

            # Diagonal matrix to vector
            case (n_row, n_col) if n_row == n_col:
                return state, init_vec_from_diag_matrix_sparse_repr(
                    child=child, shape=(n_row, 1)
                )
            
            case _:
                raise AssertionError(
                    to_operator_traceback(
                        message=f"{child.shape[0]=} is not {child.shape[1]=}",
                        stack=self.stack,
                    )
                )
