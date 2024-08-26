from abc import abstractmethod
from typing import Iterable
from typing_extensions import override

from polymat.expressiontree.expressiontree import ExpressionTree
from polymat.sparserepr.data.polynomialmatrix import MatrixIndexType
from polymat.sparserepr.data.polynomial import PolynomialType
from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.utils.getstacklines import (
    FrameSummary,
    FrameSummaryMixin,
)
from polymat.symbol import Symbol
from polymat.sparserepr.init import init_sparse_repr_from_iterable


class DefineVariableMixin(FrameSummaryMixin, ExpressionTree):
    """Underlying object for VariableExpression"""

    def __str__(self):
        return self.symbol

    @property
    @abstractmethod
    def size(self) -> int | ExpressionTree:
        """Shape of the variable expression."""

    @property
    @abstractmethod
    def symbol(self) -> Symbol:
        """The symbol representing the variable."""

    @staticmethod
    def create_variable_vector(
        state: State, variable: Symbol, size: int, stack: tuple[FrameSummary, ...]
    ):
        state, sym_index_range = state.register(
            variable,
            size=size,
            stack=stack,
        )

        def gen_polynomial_matrix() -> Iterable[tuple[MatrixIndexType, PolynomialType]]:
            for row, sym_index in enumerate(
                range(sym_index_range.start, sym_index_range.stop)
            ):
                polynomial = {((sym_index, 1),): 1.0}
                yield (row, 0), polynomial

        return state, gen_polynomial_matrix

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        if isinstance(self.size, ExpressionTree):
            state, polymat = self.size.apply(state)
            nvar = polymat.shape[0] * polymat.shape[1]
        else:
            nvar = self.size

        state, gen_polynomial_matrix = self.create_variable_vector(
            state,
            variable=self.symbol,
            size=nvar,
            stack=self.stack,
        )

        return state, init_sparse_repr_from_iterable(
            data=gen_polynomial_matrix(), shape=(nvar, 1)
        )
