from typing import Self
from dataclasses import replace
from dataclassabc import dataclassabc

from polymat.utils.getstacklines import FrameSummary, to_operator_traceback
from polymat.symbol import Symbol


@dataclassabc(frozen=True, slots=True)
class State:
    n_indices: int

    indices: dict[Symbol, tuple[int, int]]
    """ Map from variables to their indices given by a range. """

    cache: dict
    """ 
    Used to cache the computed value of an expressions (that is a SparseReprMixin object) so that
    it does not need to be recomputed again. 
    """

    def copy(self, cache: dict) -> Self:
        return replace(self, cache=cache)

    def register(
        self, 
        size: int, 
        stack: tuple[FrameSummary, ...],
        symbol: Symbol | None = None,
    ):
        """Index a variable and get its index range."""

        # symbol already exists
        if symbol in self.indices:
            start, stop = self.indices[symbol]

            if size == stop - start:
                return self, (start, stop)
            
            else:
                message = (
                    f"Symbols must be unique names! Cannot index symbol "
                    f"{symbol} with shape {size} because there is already a symbol "
                    f"with the same name with shape {(start, stop)}"
                )
                raise AssertionError(
                    to_operator_traceback(
                        message=message,
                        stack=stack,
                    )
                )

        n_indices = self.n_indices + size
        index = (self.n_indices, self.n_indices + size)

        # anonymous symbol
        if symbol is None:
            return replace(
                self,
                n_indices=n_indices,
            ), index

        return replace(
            self,
            n_indices=n_indices,
            indices=self.indices | {symbol: index},
        ), index

    # retrieval of indices
    ######################

    def _get_symbol(self, index: int):
        for symbol, (start, stop) in self.indices.items():
            if start <= index < stop:
                return symbol, (start, stop)

        raise IndexError(f"There is no variable with index {index}.")

    def get_symbol(self, index: int):
        """Get the symbol that contains the given index."""
        symbol, _ = self._get_symbol(index)
        return symbol

    def get_index_range(self, symbol: Symbol):
        return self.indices[symbol]

    def get_name(self, index: int):
        """
        Retrieve the unique name of a variable based on the provided index.

        Each variable is associated with a range of indices. This function returns a unique name corresponding to the given index.
        If a variable spans multiple indices, the base name of the variable is extended with a relative index to ensure uniqueness within that range.

        Args:
            index (int): The index corresponding to the variable whose name is being retrieved.

        Returns:
            str: The unique name of the variable associated with the specified index.
        """

        symbol, (start, stop) = self._get_symbol(index)

        # Variable is not scalar
        if stop - start > 1:
            return f"{symbol}_{index - start}"

        return str(symbol)


def init_state():
    return State(
        n_indices=0,
        indices={},
        cache={},
    )
