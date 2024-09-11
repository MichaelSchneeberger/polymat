from abc import abstractmethod
from typing import override

from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.expressiontree.nodes import SingleChildExpressionNode
from polymat.sparserepr.init import init_reshape_sparse_repr


class Reshape(SingleChildExpressionNode):
    @property
    @abstractmethod
    def new_shape(self) -> tuple[int, int]: ...

    def __str__(self):
        return f"reshape({self.child}, {self.new_shape})"

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, child = self.child.apply(state=state)

        def remaining(n_used):
            return int(child.n_entries / n_used)

        match self.new_shape:
            # replace '-1' by the remaining number of elements
            case (-1, n_used):
                shape = (remaining(n_used), n_used)

            case (n_used, -1):
                shape = (n_used, remaining(n_used))

            case _:
                shape = self.new_shape

        return state, init_reshape_sparse_repr(
            child=child,
            shape=shape,
        )
