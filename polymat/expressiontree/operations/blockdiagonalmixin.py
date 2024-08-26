from itertools import accumulate, pairwise
from typing import override

from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.expressiontree.expressiontree import MultiChildrenExpressionTreeMixin
from polymat.sparserepr.init import init_block_diagonal_sparse_repr


class BlockDiagonalMixin(MultiChildrenExpressionTreeMixin):
    def __str__(self):
        children = ",".join(str(c) for c in self.children)
        return f"block_diag({children})"

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, children = self.apply_children(state)

        def acc_row_col_pairs(acc, child):
            row, col = acc
            return row + child.shape[0], col + child.shape[1]

        row_col_pairs = accumulate(
            children,
            acc_row_col_pairs,
            initial=(0, 0),
        )

        def to_ranges(v):
            (row, col), (n_row, n_col) = v
            return range(row, n_row), range(col, n_col)

        row_col_ranges = tuple(
            map(
                to_ranges,
                pairwise(row_col_pairs),
            )
        )

        row_range, col_range = row_col_ranges[-1]

        return state, init_block_diagonal_sparse_repr(
            children=children,
            shape=(row_range.stop, col_range.stop),
            row_col_ranges=row_col_ranges,
        )
