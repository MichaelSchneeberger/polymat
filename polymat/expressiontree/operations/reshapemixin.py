from abc import abstractmethod
import functools
import operator
from typing import override

from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.expressiontree.expressiontree import SingleChildExpressionTreeMixin
from polymat.sparserepr.init import init_reshape_sparse_repr


class ReshapeMixin(SingleChildExpressionTreeMixin):
    @property
    @abstractmethod
    def new_shape(self) -> tuple[int, int]: ...

    def __str__(self):
        return f"reshape({self.child}, {self.new_shape})"

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, child = self.child.apply(state=state)

        # replace '-1' by the remaining number of elements
        if -1 in self.new_shape:
            n_total = child.shape[0] * child.shape[1]

            remaining_shape = tuple(e for e in self.new_shape if e != -1)

            assert len(remaining_shape) + 1 == len(self.new_shape)

            n_used = functools.reduce(operator.mul, remaining_shape)

            n_remaining = int(n_total / n_used)

            def gen_shape():
                for e in self.new_shape:
                    if e == -1:
                        yield n_remaining
                    else:
                        yield e

            shape = tuple(gen_shape())

        else:
            shape = self.new_shape

        return state, init_reshape_sparse_repr(
            child=child,
            shape=shape,
        )
