from abc import abstractmethod
from typing_extensions import override

from polymat.expressiontree.nodes import ExpressionNode
from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.sparserepr.init import init_sparse_repr_from_iterable


class FromVariableIndices(ExpressionNode):
    def __str__(self):
        return f"from_indices({self.indices})"

    @property
    @abstractmethod
    def indices(self) -> tuple[int, ...]:
        """The matrix of numbers in row major order."""

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        def gen_polynomial_matrix():
            for row, index in enumerate(self.indices):
                monomial = ((index, 1),)
                yield (row, 0), {monomial: 1.0}

        return state, init_sparse_repr_from_iterable(
            data=gen_polynomial_matrix(), 
            shape=(len(self.indices), 1)
        )
