from typing import Iterable

from polymat.sparserepr.data.polynomial import PolynomialType, add_polynomials


type MatrixIndexType = tuple[int, int]
type PolynomialMatrixType = dict[MatrixIndexType, PolynomialType]


def polynomial_matrix_from_iterable(
    values: Iterable[tuple[MatrixIndexType, PolynomialType]],
) -> PolynomialMatrixType:
    polymatrix = {}

    for index, polynomial in values:
        if index in polymatrix:
            summation = add_polynomials(
                left=polymatrix[index],
                right=polynomial,
            )

            if summation:
                polymatrix[index] = summation
            else:
                del polymatrix[index]
        else:
            polymatrix[index] = polynomial

    return polymatrix
