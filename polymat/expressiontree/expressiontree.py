from abc import abstractmethod
from itertools import accumulate

from statemonad.abc import StateMonadNode

from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State


class ExpressionTree(StateMonadNode[State, SparseRepr]): ...


class SingleChildExpressionTreeMixin(ExpressionTree):
    @property
    @abstractmethod
    def child(self) -> ExpressionTree: ...


class TwoChildrenExpressionTreeMixin(ExpressionTree):
    @property
    @abstractmethod
    def left(self) -> ExpressionTree: ...

    @property
    @abstractmethod
    def right(self) -> ExpressionTree: ...


class MultiChildrenExpressionTreeMixin(ExpressionTree):
    @property
    @abstractmethod
    def children(self) -> tuple[ExpressionTree, ...]: ...

    def apply_children(self, state: State):
        def acc_children(acc, next):
            state, children = acc

            state, child = next.apply(state=state)
            return state, children + (child,)

        *_, (state, children) = accumulate(
            self.children, acc_children, initial=(state, tuple())
        )

        return state, children
