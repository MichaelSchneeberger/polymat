from abc import abstractmethod
from typing import override

from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.expressiontree.expressiontree import (
    SingleChildExpressionTreeMixin,
)
from polymat.sparserepr.init import init_repmat_sparse_repr


class RepMatMixin(SingleChildExpressionTreeMixin):
    @property
    @abstractmethod
    def repetition(self) -> tuple[int, int]: ...

    def __str__(self):
        return f"rep_mat({self.child}, {self.repetition})"

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, child = self.child.apply(state=state)

        return state, init_repmat_sparse_repr(
            child=child,
            shape=tuple(s * r for s, r in zip(child.shape, self.repetition)),
            child_shape=child.shape,
        )
