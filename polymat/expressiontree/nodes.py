from abc import abstractmethod
from itertools import accumulate

from statemonad.abc import StateMonadNode

from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State


class ExpressionNode(StateMonadNode[State, SparseRepr]): ...


class SingleChildExpressionNode(
    ExpressionNode,
):
    @property
    @abstractmethod
    def child(self) -> ExpressionNode: ...


class TwoChildrenExpressionNode(
    ExpressionNode,
):
    @property
    @abstractmethod
    def left(self) -> ExpressionNode: ...

    @property
    @abstractmethod
    def right(self) -> ExpressionNode: ...


class MultiChildrenExpressionNode(ExpressionNode):
    @property
    @abstractmethod
    def children(self) -> tuple[ExpressionNode, ...]: ...

    def apply_children(self, state: State):
        def acc_children(acc, next):
            state, children = acc

            state, child = next.apply(state=state)
            return state, children + (child,)

        *_, (state, children) = accumulate(
            self.children, acc_children, initial=(state, tuple())
        )

        return state, children
