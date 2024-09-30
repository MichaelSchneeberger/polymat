from abc import abstractmethod
from functools import cached_property
import numpy as np
import scipy.sparse
import itertools

from numpy.typing import NDArray


class ArrayRepr:
    @property
    @abstractmethod
    def data(self) -> dict[int, np.ndarray]: ...

    @property
    @abstractmethod
    def n_eq(self) -> int: ...

    @property
    @abstractmethod
    def n_param(self) -> int: ...

    @property
    @abstractmethod
    def n_row(self) -> int | None: ...

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
        assert x.shape[1] == 1, f'{x} must be a numpy vector'

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

        result = sum(gen_value())

        if self.n_row:
            return np.reshape(result, (self.n_row, -1), order='F')
        else:
            return result

    @cached_property
    def degree(self) -> int:
        if self.data:
            return max(self.data.keys())
        else:
            return 0
    
    @staticmethod
    def to_column_indices(
        n_var: int,
        variable_indices: tuple[int, ...],
    ) -> set[int]:
        """
        Given a matrix storing the coefficients of the polynomial terms for a specified degree, this function
        returns the column indices corresponding to a specified monomia.

        Consider the polynomial:

            (x1 + x2 + x3)**2 = x1**2 + x1*x2 + x1*x3 + x1*x3 + x2**2 + x2*x3 + x1*x3 + x2*x3 + x3**2

        For the monomial x1*x2 (represented by variable_indices=(0,1)), the function return the set of column
        indices, such as {1, 3}, which map to the position of this monomial in the matrix.
        """
        
        variable_indices_perm = itertools.permutations(variable_indices)

        return set(
            sum(
                var_index * (n_var**index)
                for index, var_index in enumerate(monomial)
            )
            for monomial in variable_indices_perm
        )
