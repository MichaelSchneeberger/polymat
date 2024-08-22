from abc import abstractmethod
from functools import cached_property
import numpy as np
import scipy.sparse
import itertools

from numpy.typing import NDArray


class ArrayReprMixin:
    @property
    @abstractmethod
    def data(self) -> dict[int, np.ndarray]: ...

    @property
    @abstractmethod
    def n_eq(self) -> int: ...

    @property
    @abstractmethod
    def n_param(self) -> int: ...

    def __getitem__(self, degree):
        if degree not in self.data:
            # print(self.n_eq)

            if degree <= 1:
                buffer = np.zeros((self.n_eq, self.n_param**degree), dtype=np.double)

            else:
                buffer = scipy.sparse.dok_array(
                    (self.n_eq, self.n_param**degree), dtype=np.double
                )

            self.data[degree] = buffer

        return self.data[degree]

    def __str__(self):
        def gen_deg_array():
            for deg, array in self.data.items():
                if scipy.sparse.issparse(array):
                    yield deg, array.toarray()
                else:
                    yield deg, array

        return str(dict(gen_deg_array()))

    def add(self, row: int, col: int, degree: int, value: float):
        self[degree][row, col] = value

    def __call__(self, x: NDArray) -> NDArray:
        if isinstance(x, tuple) or isinstance(x, list):
            x = np.array(x).reshape(-1, 1)

        elif x.shape[0] == 1:
            x = x.reshape(-1, 1)

        def acc_x_powers(acc, _):
            next = (acc @ x.T).reshape(-1, 1)
            return next

        x_powers = tuple(
            itertools.accumulate(
                range(self.degree - 1),
                acc_x_powers,
                initial=x,
            )
        )[1:]

        def gen_value():
            for idx, equation in self.data.items():
                if idx == 0:
                    yield equation

                elif idx == 1:
                    yield equation @ x

                else:
                    yield equation @ x_powers[idx - 2]

        return sum(gen_value())

    @cached_property
    def degree(self) -> int:
        return max(self.data.keys())
