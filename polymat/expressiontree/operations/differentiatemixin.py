from abc import abstractmethod
from typing import override

from polymat.sparserepr.data.polynomial import differentiate_polynomial
from polymat.sparserepr.sparsereprmixin import SparseReprMixin
from polymat.state import State
from polymat.expressiontree.expressiontreemixin import (
    ExpressionTreeMixin,
    SingleChildExpressionTreeMixin,
)
from polymat.utils.getstacklines import FrameSummaryMixin, to_operator_exception
from polymat.sparserepr.init import init_sparse_repr_from_data


class DifferentiateMixin(FrameSummaryMixin, SingleChildExpressionTreeMixin):
    @property
    @abstractmethod
    def variables(self) -> ExpressionTreeMixin: ...

    def __str__(self):
        return f"diff({self.child}, {self.variables})"

    @override
    def apply(self, state: State) -> tuple[State, SparseReprMixin]:
        state, child = self.child.apply(state=state)
        state, variable_vector = self.variables.apply(state=state)

        if not (child.shape[1] == 1):
            raise AssertionError(
                to_operator_exception(
                    message=f"{child.shape[1]=} is not 1",
                    stack=self.stack,
                )
            )

        # keep order of variable indices
        indices = tuple(variable_vector.to_indices())

        def gen_polynomial_matrix():
            for row in range(child.shape[0]):
                polynomial = child.at(row, 0)

                if polynomial:
                    for col, index in enumerate(indices):
                        derivative = differentiate_polynomial(polynomial, index)

                        if derivative:
                            yield (row, col), derivative

        data = dict(gen_polynomial_matrix())

        return state, init_sparse_repr_from_data(
            data=data,
            shape=(child.shape[0], len(indices)),
        )
