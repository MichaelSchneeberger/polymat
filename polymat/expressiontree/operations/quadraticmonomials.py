import abc
from typing import override

from polymat.expressiontree.nodes import (
    ExpressionNode,
    SingleChildExpressionNode,
)
from polymat.sparserepr.data.monomial import sort_monomials, split_monomial_indices
from polymat.sparserepr.init import init_sparse_repr_from_iterable
from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State


class QuadraticMonomials(SingleChildExpressionNode):
    # FIXME: docstring, what does this thing even do
    """
    Maps a polynomial matrix

        underlying = [
            [x y    ],
            [x + x^2],
        ]

    into a vector of monomials

        output = [1, x, y]

    in variable

        variables = [x, y].
    """

    @property
    @abc.abstractmethod
    def variables(self) -> ExpressionNode: ...

    def __str__(self):
        return f"quadratic_monomials_in({self.child}, {self.variables})"

    # overwrites the abstract method of `ExpressionBaseMixin`
    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, child = self.child.apply(state=state)
        state, variable_vector = self.variables.apply(state=state)

        indices = set(variable_vector.to_indices())

        def gen_linear_monomials():
            for _, polynomial in child.entries():
                for monomial in polynomial.keys():
                    x_monomials = tuple(
                        (index, power) for index, power in monomial if index in indices
                    )

                    left_monomials, right_monomials = split_monomial_indices(
                        x_monomials
                    )

                    yield left_monomials
                    yield right_monomials

        # sort monomials for clearer visual representation in the output
        sorted_monomials = sort_monomials(set(gen_linear_monomials()))

        def gen_polynomial_matrix():
            for index, monomial in enumerate(sorted_monomials):
                yield (index, 0), {monomial: 1.0}

        polymatrix = init_sparse_repr_from_iterable(
            data=gen_polynomial_matrix(),
            shape=(len(sorted_monomials), 1),
        )

        return state, polymatrix
