from numpy.typing import NDArray

from dataclassabc import dataclassabc

from polymat.arrayrepr.arrayrepr import ArrayRepr


@dataclassabc(frozen=True)
class ArrayReprImpl(ArrayRepr):
    data: dict[int, NDArray]
    n_eq: int
    n_param: int
    n_row: int | None


def init_array_repr(
    n_eq: int,
    n_param: int,
    n_row: int | None = None,
):
    return ArrayReprImpl(
        data={},
        n_eq=n_eq,
        n_param=n_param,
        n_row=n_row,
    )
