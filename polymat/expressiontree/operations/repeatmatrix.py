from abc import abstractmethod
from typing import override

from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.expressiontree.nodes import (
    SingleChildExpressionNode,
)
from polymat.sparserepr.init import init_repmat_sparse_repr


class RepeatMatrix(SingleChildExpressionNode):
    @property
    @abstractmethod
    def repetition(self) -> tuple[int, int]: ...

    def __str__(self):
        return f"rep_mat({self.child}, {self.repetition})"

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, child = self.child.apply(state=state)

        n_rows, n_cols = child.shape
        row_rep, col_rep = self.repetition

        return state, init_repmat_sparse_repr(
            child=child,
            shape=(n_rows * row_rep, n_cols * col_rep),
            child_shape=child.shape,
        )
