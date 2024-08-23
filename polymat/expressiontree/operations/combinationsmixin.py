import abc

from itertools import combinations_with_replacement

from polymat.sparserepr.data.polynomial import (
    constant_polynomial,
    multiply_polynomial_iterable,
)
from polymat.utils.getstacklines import FrameSummaryMixin, to_operator_traceback
from polymat.expressiontree.expressiontreemixin import SingleChildExpressionTreeMixin
from polymat.sparserepr.sparsereprmixin import SparseReprMixin
from polymat.state import State
from polymat.sparserepr.init import init_sparse_repr_from_iterable


class CombinationsMixin(FrameSummaryMixin, SingleChildExpressionTreeMixin):
    # FIXME: improve docstring
    """
    combination using degrees=(0, 1, 2):

    [[x], [y]]  ->  [[1], [x], [y], [x**2], [x*y], [y**2]]
    """

    def __str__(self):
        match self.degrees:
            case tuple():
                return f"combinations({self.child}, degrees={self.degrees})"
            case _:
                return f"combinations({self.child})"

    @property
    @abc.abstractmethod
    def degrees(self) -> tuple[int, ...]:
        """
        Vector or scalar expression, or a list of integers.
        """

    def apply(
        self,
        state: State,
    ) -> tuple[State, SparseReprMixin]:
        state, child = self.child.apply(state)

        if not (child.shape[1] == 1):
            raise AssertionError(
                to_operator_traceback(
                    message=f"{child.shape[1]=} is not 1",
                    stack=self.stack,
                )
            )

        def gen_combinations():
            for degree in self.degrees:
                yield from combinations_with_replacement(range(child.shape[0]), degree)

        combinations = tuple(gen_combinations())

        def gen_polynomial_matrix():
            for row, combination in enumerate(combinations):
                index = (row, 0)

                # x.combinations((0, 1, 2)) produces [1, x, x**2]
                if len(combination) == 0:
                    yield index, constant_polynomial(1.0)
                    continue

                polynomials = (child.at(row, 0) for row in combination)

                result = multiply_polynomial_iterable(polynomials)

                if result:
                    yield index, result

        return state, init_sparse_repr_from_iterable(
            data=gen_polynomial_matrix(),
            shape=(len(combinations), 1),
        )
