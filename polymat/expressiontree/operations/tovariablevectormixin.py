from typing import override

from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.expressiontree.expressiontree import SingleChildExpressionTreeMixin
from polymat.sparserepr.init import init_sparse_repr_from_iterable


class ToVariableVectorMixin(SingleChildExpressionTreeMixin):
    def __str__(self):
        return f"to_variable_vector({self.child})"

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, child = self.child.apply(state=state)

        sorted_indices = sorted(set(child.to_indices()))

        def gen_polynomial_matrix():
            for row, index in enumerate(sorted_indices):
                variable_monomial = ((index, 1),)
                yield (row, 0), {variable_monomial: 1.0}

        return state, init_sparse_repr_from_iterable(
            data=gen_polynomial_matrix(),
            shape=(len(sorted_indices), 1),
        )
