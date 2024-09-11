import abc
from typing import override

from polymat.expressiontree.nodes import (
    SingleChildExpressionNode,
)
from polymat.expressiontree.data.variables import VariableType, to_indices
from polymat.sparserepr.init import init_sparse_repr_from_iterable
from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State


class TruncateMonomials(SingleChildExpressionNode):
    DegreeType = tuple[int, ...]

    @property
    @abc.abstractmethod
    def variables(self) -> VariableType: ...

    @property
    @abc.abstractmethod
    def degrees(self) -> DegreeType: ...

    def __str__(self):
        return f"truncate_monomials({self.child}, {self.variables}, {self.degrees})"

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, child = self.child.apply(state=state)
        state, indices = to_indices(state, self.variables)

        def gen_polymatrix():
            for matrix_index, polynomial in child.entries():

                def gen_truncated_polynomial():
                    for monomial, value in polynomial.items():
                        degree = sum(
                            (count for index, count in monomial if index in indices)
                        )

                        if degree in self.degrees:
                            yield monomial, value

                result = dict(gen_truncated_polynomial())

                if result:
                    yield matrix_index, result

        polymatrix = init_sparse_repr_from_iterable(
            data=gen_polymatrix(),
            shape=child.shape,
        )

        return state, polymatrix
