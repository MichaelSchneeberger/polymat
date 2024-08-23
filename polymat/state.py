from typing import NamedTuple, Self
from dataclasses import replace
from dataclassabc import dataclassabc

from polymat.utils.getstacklines import FrameSummary, to_operator_traceback
from polymat.variable import Variable


@dataclassabc(frozen=True)
class State:
    class IndexRange(NamedTuple):
        start: int
        """ Start of the indices """

        stop: int
        """ End of the indices, this value is not included """

        def __len__(self) -> int:
            return self.stop - self.start

        def __iter__(self):
            return iter(range(self.start, self.stop))

    n_indices: int

    indices: dict[Variable, IndexRange]
    """ Map from variables to their indices given by a range. """

    cache: dict
    """ 
    Used to cache the computed value of an expressions (that is a SparseReprMixin object) so that
    it does not need to be recomputed again. 
    """

    def copy(self, cache: dict) -> Self:
        return replace(self, cache=cache)

    def register(
        self, variable: Variable, size: int, stack: tuple[FrameSummary, ...]
    ) -> tuple[Self, IndexRange]:
        """Index a variable and get its index range."""

        if variable in self.indices:
            irange = self.indices[variable]
            if size == irange.stop - irange.start:
                return self, irange
            else:
                message = (
                    f"Symbols must be unique names! Cannot index symbol "
                    f"{variable} with shape {size} because there is already a symbol "
                    f"with the same name with shape {(irange.start, irange.stop)}"
                )
                raise AssertionError(
                    to_operator_traceback(
                        message=message,
                        stack=stack,
                    )
                )

        # If not save new index
        index = State.IndexRange(
            start=self.n_indices,
            stop=self.n_indices + size,
        )

        return replace(
            self,
            n_indices=self.n_indices + size,
            indices=self.indices | {variable: index},
        ), index

    # retrieval of indices
    ######################

    def _get_variable(self, index: int) -> tuple[Variable, IndexRange]:
        for variable, index_range in self.indices.items():
            if index_range.start <= index < index_range.stop:
                return variable, index_range

        raise IndexError(f"There is no variable with index {index}.")

    def get_variable(self, index: int) -> Variable:
        """Get the symbol that contains the given index."""
        return self._get_variable(index)[0]

    def get_index_range(self, variable: Variable):
        return self.indices[variable]

    def get_name(self, index: int) -> str:
        """
        Retrieve the unique name of a variable based on the provided index.

        Each variable is associated with a range of indices. This function returns a unique name corresponding to the given index.
        If a variable spans multiple indices, the base name of the variable is extended with a relative index to ensure uniqueness within that range.

        Args:
            index (int): The index corresponding to the variable whose name is being retrieved.

        Returns:
            str: The unique name of the variable associated with the specified index.
        """

        variable, index_range = self._get_variable(index)

        # Variable is not scalar
        if index_range.stop - index_range.start > 1:
            return f"{variable}_{index - index_range.start}"

        return str(variable)


def init_state():
    return State(
        n_indices=0,
        indices={},
        cache={},
    )
