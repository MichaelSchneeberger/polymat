import abc
from typing import override

from polymat.expressiontree.data.variables import VariableType, to_indices
from polymat.expressiontree.nodes import (
    ExpressionNode,
    SingleChildExpressionNode,
)
from polymat.sparserepr.init import init_sparse_repr_from_iterable
from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.utils.getstacklines import (
    FrameSummaryMixin,
    to_operator_traceback,
)


class LinearCoefficients(FrameSummaryMixin, SingleChildExpressionNode):
    """
    Maps a polynomial column vector

        underlying = [
            [1 + a x],
            [x^2    ],
        ]

    into a polynomial matrix

        output = [
            [1, a, 0],
            [0, 0, 1],
        ],

    where each column corresponds to a monomial defined by

        monomials = [1, x, x^2].
    """

    @property
    @abc.abstractmethod
    def monomials(self) -> ExpressionNode: ...

    @property
    @abc.abstractmethod
    def variables(self) -> VariableType: ...

    @property
    @abc.abstractmethod
    def ignore_unmatched(self) -> bool: ...

    def __str__(self):
        return f"linear_in({self.child}, {self.variables})"

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, child = self.child.apply(state=state)
        state, monomial_vector = self.monomials.apply(state=state)
        state, indices = to_indices(state, self.variables)

        if not (child.shape[1] == 1):
            raise AssertionError(
                to_operator_traceback(
                    message=f"{child.shape[1]=} is not 1",
                    stack=self.stack,
                )
            )

        # keep order of monomials
        monomials = tuple(monomial_vector.to_monomials())

        def gen_polymatrix():
            for row in range(child.shape[0]):
                polynomial = child.at(row, 0)

                if polynomial is None:
                    continue

                for monomial, value in polynomial.items():
                    x_monomial = tuple(
                        (index, power) for index, power in monomial if index in indices
                    )
                    p_monomial = tuple(
                        (index, power)
                        for index, power in monomial
                        if index not in indices
                    )

                    try:
                        col = monomials.index(x_monomial)
                    except ValueError:
                        if self.ignore_unmatched:
                            continue
                        else:
                            raise Exception(f"{x_monomial} not in {monomial_vector}")

                    yield (row, col), {p_monomial: value}

        polymatrix = init_sparse_repr_from_iterable(
            data=gen_polymatrix(),
            shape=(child.shape[0], monomial_vector.shape[0]),
        )

        return state, polymatrix
