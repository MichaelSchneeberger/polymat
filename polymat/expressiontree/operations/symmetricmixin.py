from typing_extensions import override

from polymat.sparserepr.init import init_symmetric_sparse_repr
from polymat.sparserepr.sparsereprmixin import SparseReprMixin
from polymat.state import State
from polymat.expressiontree.expressiontreemixin import SingleChildExpressionTreeMixin


class SymmetricMixin(SingleChildExpressionTreeMixin):
    def __str__(self):
        return f"symmetric({self.child})"

    @override
    def apply(self, state: State) -> tuple[State, SparseReprMixin]:
        state, child = self.child.apply(state)

        polymatrix = init_symmetric_sparse_repr(child=child)

        return state, polymatrix
