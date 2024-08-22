from typing import override

from polymat.sparserepr.sparsereprmixin import SparseReprMixin
from polymat.state import State
from polymat.expressiontree.expressiontreemixin import TwoChildrenExpressionTreeMixin
from polymat.sparserepr.init import init_kron_sparse_repr


class KronMixin(TwoChildrenExpressionTreeMixin):
    def __str__(self):
        return f"kron({self.left}, {self.right})"

    @override
    def apply(self, state: State) -> tuple[State, SparseReprMixin]:
        state, left = self.left.apply(state)
        state, right = self.right.apply(state)

        return state, init_kron_sparse_repr(
            left=left,
            right=right,
            shape=(left.shape[0] * right.shape[0], left.shape[1] * right.shape[1]),
        )
