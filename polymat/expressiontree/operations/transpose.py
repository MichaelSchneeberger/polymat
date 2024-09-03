from typing import override

from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.expressiontree.nodes import SingleChildExpressionNode
from polymat.sparserepr.init import init_transpose_sparse_repr


class Transpose(SingleChildExpressionNode):
    def __str__(self):
        return f"{self.child}.T"

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, child = self.child.apply(state=state)

        return state, init_transpose_sparse_repr(
            child=child,
        )
