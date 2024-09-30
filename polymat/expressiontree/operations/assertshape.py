from abc import abstractmethod
from typing import Callable, override

from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.expressiontree.nodes import SingleChildExpressionNode
from polymat.utils.getstacklines import FrameSummaryMixin, to_operator_traceback


class AssertShape(FrameSummaryMixin, SingleChildExpressionNode):
    AssertionType = Callable[[int, int], bool]
    MessageType = Callable[[int, int], str]

    @property
    @abstractmethod
    def func(self) -> AssertionType: ...

    @property
    @abstractmethod
    def message(self) -> MessageType: ...

    def __str__(self):
        return str(self.child)

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, child = self.child.apply(state=state)

        if not self.func(*child.shape):
            raise AssertionError(
                to_operator_traceback(
                    message=self.message(*child.shape),
                    stack=self.stack,
                )
            )

        return state, child
