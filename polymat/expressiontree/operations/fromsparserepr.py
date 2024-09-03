from abc import abstractmethod
from typing_extensions import override

from polymat.state import State
from polymat.expressiontree.nodes import ExpressionNode
from polymat.sparserepr.sparserepr import SparseRepr


class FromSparseRepr(ExpressionNode):
    """
    Make an expression from a tuple of tuples of numbers (constant). The tuple
    of tuples is interpreted as a matrix stored with row major ordering.

    ..code:: py
        m = polymatrix.from_numbers(((0, 1), (1, 0))
    """

    def __str__(self):
        return str(self.sparse_repr)

    @property
    @abstractmethod
    def sparse_repr(self) -> SparseRepr:
        """The matrix of numbers in row major order."""

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        return state, self.sparse_repr
