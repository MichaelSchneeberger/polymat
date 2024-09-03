import math

from abc import abstractmethod
from typing_extensions import override
from numpy.typing import NDArray

from polymat.expressiontree.nodes import ExpressionNode
from polymat.sparserepr.data.polynomial import constant_polynomial
from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.sparserepr.init import init_sparse_repr_from_iterable


class FromNumpy(ExpressionNode):
    """
    Make a (constant) expression from a numpy array.

    ..code:: py
        identity = polymatrix.from_numpy(np.eye(3))
    """

    def __str__(self):
        return f"from_numpy({self.data})"

    @property
    @abstractmethod
    def data(self) -> NDArray:
        """The Numpy array."""

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        if len(self.data.shape) > 2:
            raise ValueError(
                "Cannot construct expression from numpy array with "
                f"shape {self.data.shape}, only vectors and matrices are allowed"
            )

        data = self.data

        nrows, ncols = data.shape

        def gen_polynomial_matrix():
            for row in range(nrows):
                for col in range(ncols):
                    if not math.isclose(data[row, col], 0):
                        yield (row, col), constant_polynomial(data[row, col])

        return state, init_sparse_repr_from_iterable(
            data=gen_polynomial_matrix(), shape=(nrows, ncols)
        )
