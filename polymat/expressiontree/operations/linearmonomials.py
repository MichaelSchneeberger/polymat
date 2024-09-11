import abc
from typing import override

from polymat.expressiontree.data.variables import VariableType, to_indices
from polymat.expressiontree.nodes import (
    SingleChildExpressionNode,
)
from polymat.sparserepr.data.monomial import sort_monomials
from polymat.sparserepr.init import init_sparse_repr_from_iterable
from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State


class LinearMonomials(SingleChildExpressionNode):
    """
    Maps a polynomial matrix

        underlying = [
            [1,   a x    ],
            [x^2, x + x^2],
        ]

    into a vector of monomials

        output = [1, x, x^2]

    in variable

        variables = [x].
    """

    @property
    @abc.abstractmethod
    def variables(self) -> VariableType: ...

    def __str__(self):
        return f"linear_monomials_in({self.child}, {self.variables})"

    # overwrites the abstract method of `ExpressionBaseMixin`
    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, child = self.child.apply(state=state)
        state, indices = to_indices(state, self.variables)

        def gen_linear_monomials():
            for _, polynomial in child.entries():
                for monomial in polynomial.keys():
                    x_monomial = tuple(
                        (index, power) for index, power in monomial if index in indices
                    )

                    yield x_monomial

        sorted_monomials = sort_monomials(set(gen_linear_monomials()))

        def gen_polynomial_matrix():
            for index, monomial in enumerate(sorted_monomials):
                yield (index, 0), {monomial: 1.0}

        polymatrix = init_sparse_repr_from_iterable(
            data=gen_polynomial_matrix(),
            shape=(len(sorted_monomials), 1),
        )

        return state, polymatrix
