from abc import abstractmethod
from typing import override

from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.expressiontree.nodes import (
    SingleChildExpressionNode,
)
from polymat.sparserepr.init import init_get_item_sparse_repr


class GetItem(SingleChildExpressionNode):
    KeyValueType = int | slice | tuple[int, ...]
    KeyType = tuple[KeyValueType, KeyValueType]

    @property
    @abstractmethod
    def key(self) -> KeyType:
        """The slice."""
        # Type / format of this property must match of slice accepted by
        # SlicePolyMatrix, since it directly uses that see
        # polymatrix.polymatrix.init.init_poly_matrix

        # TODO: allow slice to be an Expression that evaluates to a number or
        # vector of numbers

    def __str__(self):
        return f"slice({self.child}, {self.key})"

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, child = self.child.apply(state=state)

        def format_key(state: State, key: GetItem.KeyValueType):
            match key:
                case int():
                    fkey = (key,)
                case tuple():
                    fkey = key
                case slice(start=start, stop=stop):
                    fkey = tuple(range(start, stop))

            return state, fkey

        state, row_key = format_key(state, self.key[0])
        state, col_key = format_key(state, self.key[1])

        return state, init_get_item_sparse_repr(
            child=child,
            shape=(len(row_key), len(col_key)),
            key=(row_key, col_key),
        )
