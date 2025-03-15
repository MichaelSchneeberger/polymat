from abc import abstractmethod
from itertools import accumulate

from statemonad.abc import StateMonadNode

from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state.state import State as BaseState


class ExpressionNode[State: BaseState](StateMonadNode[State, SparseRepr]): ...


class SingleChildExpressionNode[State: BaseState](
    ExpressionNode[State],
):
    @property
    @abstractmethod
    def child(self) -> ExpressionNode[State]: ...


class TwoChildrenExpressionNode[State: BaseState](
    ExpressionNode[State],
):
    @property
    @abstractmethod
    def left(self) -> ExpressionNode[State]: ...

    @property
    @abstractmethod
    def right(self) -> ExpressionNode[State]: ...


class MultiChildrenExpressionNode[State: BaseState](
    ExpressionNode[State],
):
    @property
    @abstractmethod
    def children(self) -> tuple[ExpressionNode[State], ...]: ...

    def apply_children(self, state: State):
        def acc_children(acc, next):
            state, children = acc

            state, child = next.apply(state=state)
            return state, children + (child,)

        *_, (state, children) = accumulate(
            self.children, acc_children, initial=(state, tuple())
        )

        return state, children
