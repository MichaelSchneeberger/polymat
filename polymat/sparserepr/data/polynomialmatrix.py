from typing import Iterable

from polymat.sparserepr.data.polynomial import PolynomialType, add_polynomials


type MatrixIndexType = tuple[int, int]
type PolynomialMatrixType = dict[MatrixIndexType, PolynomialType]


def add_polynomial_to_polynomial_matrix_mutable(
    mutable: PolynomialMatrixType,
    index: MatrixIndexType,
    polynomial: PolynomialType,
):
    if index in mutable:
        summation = add_polynomials(
            left=mutable[index],
            right=polynomial,
        )

        if summation:
            mutable[index] = summation
        else:
            del mutable[index]
    else:
        mutable[index] = polynomial


def polynomial_matrix_from_iterable(
    values: Iterable[tuple[MatrixIndexType, PolynomialType]],
) -> PolynomialMatrixType:
    polymatrix = {}

    for index, polynomial in values:
        add_polynomial_to_polynomial_matrix_mutable(
            mutable=polymatrix,
            index=index,
            polynomial=polynomial,
        )

    return polymatrix
