from abc import abstractmethod
from itertools import accumulate

from statemonad.abc import StateMonadNode

from polymat.sparserepr.sparsereprmixin import SparseReprMixin
from polymat.state import State


class ExpressionTreeMixin(StateMonadNode[State, SparseReprMixin]): ...


class SingleChildExpressionTreeMixin(ExpressionTreeMixin):
    @property
    @abstractmethod
    def child(self) -> ExpressionTreeMixin: ...


class TwoChildrenExpressionTreeMixin(ExpressionTreeMixin):
    @property
    @abstractmethod
    def left(self) -> ExpressionTreeMixin: ...

    @property
    @abstractmethod
    def right(self) -> ExpressionTreeMixin: ...


class MultiChildrenExpressionTreeMixin(ExpressionTreeMixin):
    @property
    @abstractmethod
    def children(self) -> tuple[ExpressionTreeMixin, ...]: ...

    def apply_children(self, state: State):
        def acc_children(acc, next):
            state, children = acc

            state, child = next.apply(state=state)
            return state, children + (child,)

        *_, (state, children) = accumulate(
            self.children, acc_children, initial=(state, tuple())
        )

        return state, children
