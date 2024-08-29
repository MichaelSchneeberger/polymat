from typing import override

from polymat.expressiontree.nodes import TwoChildrenExpressionNode
from polymat.sparserepr.data.polynomial import (
    add_polynomial_iterable,
    multiply_polynomials,
)
from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.utils.getstacklines import FrameSummaryMixin, to_operator_traceback
from polymat.sparserepr.init import init_sparse_repr_from_iterable


class MatrixMultMixin(FrameSummaryMixin, TwoChildrenExpressionNode):
    def __str__(self):
        return f"matmul({self.left}, {self.right})"

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, left = self.left.apply(state=state)
        state, right = self.right.apply(state=state)

        if not (left.shape[1] == right.shape[0]):
            msg = (
                f"Cannot multiply matrices {self.left} and {self.right} because their shapes "
                f"{left.shape}, and {right.shape} do not match!"
            )
            raise AssertionError(
                to_operator_traceback(
                    message=msg,
                    stack=self.stack,
                )
            )

        def gen_polynomial_matrix():
            for row in range(left.shape[0]):
                for col in range(right.shape[1]):

                    def gen_polynomials():
                        for k in range(left.shape[1]):
                            left_polynomial = left.at(row, k)
                            right_polynomial = right.at(k, col)

                            if left_polynomial and right_polynomial:
                                result = multiply_polynomials(
                                    left_polynomial, right_polynomial
                                )
                                if result:
                                    yield result

                    summation = add_polynomial_iterable(gen_polynomials())

                    if summation:
                        yield (row, col), summation

        return state, init_sparse_repr_from_iterable(
            gen_polynomial_matrix(), shape=(left.shape[0], right.shape[1])
        )
