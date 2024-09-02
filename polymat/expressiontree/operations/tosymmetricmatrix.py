from typing import override

from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.expressiontree.nodes import SingleChildExpressionNode
from polymat.sparserepr.init import init_sparse_repr_from_iterable, init_symmetric_sparse_repr, init_transpose_sparse_repr
from polymat.utils.getstacklines import FrameSummaryMixin, to_operator_traceback


class ToSymmetricMatrix(FrameSummaryMixin, SingleChildExpressionNode):
    def __str__(self):
        return f"to_symmetric_matrix({self.child})"

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, child = self.child.apply(state=state)

        def from_vector_to_symmetric_matrix(child: SparseRepr):
            def invert_binomial_coefficient(val):
                idx = 1
                sum_val = 1

                while sum_val < val:
                    idx += 1
                    sum_val += idx

                assert sum_val == val, f"{sum_val=} is not equal {val=}"

                return idx

            n_rows = invert_binomial_coefficient(child.shape[0])

            def gen_polynomial_matrix():
                v_row = 0

                for m_row in range(n_row):
                    for m_col in range(m_row, n_row):
                        polynomial = child.at(v_row, 0)

                        if polynomial:
                            yield (m_row, m_col), polynomial

                        v_row += 1

            sparse_repr = init_sparse_repr_from_iterable(
                gen_polynomial_matrix(), shape=(n_rows, n_rows)
            )

            return init_symmetric_sparse_repr(
                child=sparse_repr,
            )

        match child.shape:

            # Vector to diagonal matrix
            case (int(), 1):
                return state, from_vector_to_symmetric_matrix(child)
            
            case (1, int()):
                return state, from_vector_to_symmetric_matrix(
                    child=init_transpose_sparse_repr(child),
                )

            # Diagonal matrix to vector
            case (n_row, n_col) if n_row == n_col:
                return state, child
            
            case _:
                raise AssertionError(
                    to_operator_traceback(
                        message=f"{child.shape[0]=} is not {child.shape[1]=}",
                        stack=self.stack,
                    )
                )
