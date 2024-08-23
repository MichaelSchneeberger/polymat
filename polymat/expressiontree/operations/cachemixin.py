from typing import override

from polymat.sparserepr.init import init_sparse_repr_from_data
from polymat.sparserepr.operations.sparsereprfrompolynomialmixin import (
    SparseReprFromPolynomialMatrixMixin,
)
from polymat.sparserepr.sparsereprmixin import SparseReprMixin
from polymat.state import State
from polymat.expressiontree.expressiontreemixin import SingleChildExpressionTreeMixin
from polymat.utils.getstacklines import FrameSummaryMixin, to_operator_traceback


class CacheMixin(FrameSummaryMixin, SingleChildExpressionTreeMixin):
    """Caches the polynomial matrix using the state"""

    def __str__(self):
        return str(self.child)

    @override
    def apply(self, state: State) -> tuple[State, SparseReprMixin]:
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

        if isinstance(child, SparseReprFromPolynomialMatrixMixin):
            cached_data = child.data
        else:
            cached_data = dict(child.entries())

        polymatrix = init_sparse_repr_from_data(
            data=cached_data,
            shape=child.shape,
        )

        state = state.copy(cache=state.cache | {self: polymatrix})

        return state, polymatrix
