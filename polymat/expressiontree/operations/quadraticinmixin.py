import abc
from typing import override

from polymat.expressiontree.expressiontree import (
    ExpressionTree,
    SingleChildExpressionTreeMixin,
)
from polymat.sparserepr.data.monomial import split_monomial_indices
from polymat.sparserepr.init import init_sparse_repr_from_iterable
from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.utils.getstacklines import FrameSummaryMixin, to_operator_traceback


class QuadraticInExprMixin(FrameSummaryMixin, SingleChildExpressionTreeMixin):
    @property
    @abc.abstractmethod
    def monomials(self) -> ExpressionTree: ...

    @property
    @abc.abstractmethod
    def variables(self) -> ExpressionTree: ...

    @property
    @abc.abstractmethod
    def ignore_unmatched(self) -> bool: ...

    def __str__(self):
        return f"quadratic_in({self.child}, {self.variables})"

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, child = self.child.apply(state=state)
        state, monomial_vector = self.monomials.apply(state=state)
        state, variable_vector = self.variables.apply(state=state)

        if not (child.shape[1] == 1):
            raise AssertionError(
                to_operator_traceback(
                    message=f"{child.shape[1]=} is not 1",
                    stack=self.stack,
                )
            )

        # keep order of monomials
        monomials = tuple(monomial_vector.to_monomials())

        indices = set(variable_vector.to_indices())

        for row in range(child.shape[0]):
            polynomial = child.at(row, 0)

            if polynomial is None:
                continue

            def gen_polymatrix():
                for monomial, value in polynomial.items():  # type: ignore
                    x_monomial = tuple(
                        (index, count) for index, count in monomial if index in indices
                    )
                    p_monomial = tuple(
                        (index, count)
                        for index, count in monomial
                        if index not in indices
                    )

                    left, right = split_monomial_indices(x_monomial)

                    try:
                        col = monomials.index(left)
                    except ValueError:
                        raise AssertionError(
                            to_operator_traceback(
                                message=f"{left=} not in {monomials}",
                                stack=self.stack,
                            )
                        )

                    try:
                        row = monomials.index(right)
                    except ValueError:
                        raise AssertionError(
                            to_operator_traceback(
                                message=f"{right=} not in {monomials}",
                                stack=self.stack,
                            )
                        )

                    yield (row, col), {p_monomial: value}

        polymatrix = init_sparse_repr_from_iterable(
            data=gen_polymatrix(),
            shape=(monomial_vector.shape[0], monomial_vector.shape[0]),
        )

        return state, polymatrix
