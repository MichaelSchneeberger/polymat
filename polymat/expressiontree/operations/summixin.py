from typing import override

from polymat.sparserepr.data.polynomial import add_polynomial_iterable
from polymat.sparserepr.init import init_transpose_sparse_repr
from polymat.sparserepr.sparsereprmixin import SparseReprMixin
from polymat.state import State
from polymat.expressiontree.expressiontreemixin import SingleChildExpressionTreeMixin
from polymat.sparserepr.init import init_sparse_repr_from_iterable


class SumMixin(SingleChildExpressionTreeMixin):
    """
    For each row of the matrix sum the colum elements.

    [[1, 2, 3],      [[6],
     [4, 5, 6]]  ->   [15]]
    """

    def __str__(self):
        return f"sum({self.child})"

    @override
    def apply(self, state: State) -> tuple[State, SparseReprMixin]:
        state, child = self.child.apply(state=state)

        if child.shape[1] == 1:
            # sum elements of column vector
            child = init_transpose_sparse_repr(child=child)

        def gen_polynomial_matrix():
            for row in range(child.shape[0]):
                result = {}

                def gen_polynomials():
                    for col in range(child.shape[1]):
                        polynomial = child.at(row, col)

                        if polynomial:
                            yield polynomial


                result = add_polynomial_iterable(gen_polynomials())

                if result:
                    yield ((row, 0), result)

        return state, init_sparse_repr_from_iterable(
            data=gen_polynomial_matrix(), shape=(child.shape[0], 1)
        )
