from typing import override

from polymat.sparserepr.init import init_from_polynomial_matrix
from polymat.sparserepr.operations.frompolynomialmixin import (
    FromPolynomialMatrixMixin,
)
from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.expressiontree.nodes import SingleChildExpressionNode
from polymat.utils.getstacklines import FrameSummaryMixin, to_operator_traceback


class Cache(FrameSummaryMixin, SingleChildExpressionNode):
    """Caches the polynomial matrix using the state"""

    def __str__(self):
        return str(self.child)

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        try:
            if self in state.cache:
                return state, state.cache[self]
        except TypeError:
            raise TypeError(
                to_operator_traceback(
                    message="unhashable polynomial expression",
                    stack=self.stack,
                )
            )

        state, child = self.child.apply(state)

        if isinstance(child, FromPolynomialMatrixMixin):
            cached_data = child.data
        else:
            cached_data = dict(child.entries())

        polymatrix = init_from_polynomial_matrix(
            data=cached_data,
            shape=child.shape,
        )

        state = state.copy(cache=state.cache | {self: polymatrix})

        return state, polymatrix
