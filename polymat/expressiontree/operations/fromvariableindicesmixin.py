from abc import abstractmethod
from typing_extensions import override

from polymat.expressiontree.expressiontreemixin import ExpressionTreeMixin
from polymat.sparserepr.sparsereprmixin import SparseReprMixin
from polymat.state import State
from polymat.sparserepr.init import init_sparse_repr_from_iterable


class FromVariableIndicesMixin(ExpressionTreeMixin):
    def __str__(self):
        return f"from_indices({self.indices})"

    @property
    @abstractmethod
    def indices(self) -> tuple[int, ...]:
        """The matrix of numbers in row major order."""

    @override
    def apply(self, state: State) -> tuple[State, SparseReprMixin]:
        def gen_polynomial_matrix():
            for row, index in enumerate(self.indices):
                monomial = ((index, 1),)
                yield (row, 0), {monomial: 1.0}

        shape = (len(self.indices), 1)
        return state, init_sparse_repr_from_iterable(
            data=gen_polynomial_matrix(), shape=shape
        )
