from typing import override

from polymat.sparserepr.sparsereprmixin import SparseReprMixin
from polymat.state import State
from polymat.expressiontree.expressiontreemixin import SingleChildExpressionTreeMixin
from polymat.sparserepr.init import init_transpose_sparse_repr


class TransposeMixin(SingleChildExpressionTreeMixin):
    def __str__(self):
        return f"{self.child}.T"

    @override
    def apply(self, state: State) -> tuple[State, SparseReprMixin]:
        state, child = self.child.apply(state=state)

        return state, init_transpose_sparse_repr(
            child=child,
        )
