import itertools

from typing import override

from polymat.utils.getstacklines import FrameSummaryMixin, to_operator_traceback
from polymat.sparserepr.sparserepr import SparseRepr
from polymat.sparserepr.init import init_vstack_sparse_repr
from polymat.state import State
from polymat.expressiontree.expressiontree import MultiChildrenExpressionTreeMixin


class VStackMixin(FrameSummaryMixin, MultiChildrenExpressionTreeMixin):
    def __str__(self):
        children = ",".join(str(c) for c in self.children)
        return f"v_stack({children})"

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, children = self.apply_children(state)

        n_col = children[0].shape[1]
        for child in children[1:]:
            if not (child.shape[1] == n_col):
                raise AssertionError(
                    to_operator_traceback(
                        message=f"{child.shape[1]} not equal {n_col}",
                        stack=self.stack,
                    )
                )

        row_ranges = tuple(
            range(a, b)
            for a, b in itertools.pairwise(
                itertools.accumulate((child.shape[0] for child in children), initial=0)
            )
        )

        n_row = row_ranges[-1].stop

        return state, init_vstack_sparse_repr(
            children=children,
            row_ranges=row_ranges,
            shape=(n_row, n_col),
        )
