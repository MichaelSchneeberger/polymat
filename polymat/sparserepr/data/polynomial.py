import itertools
import math
from typing import Iterable

from polymat.sparserepr.data.monomial import (
    MonomialType,
    add_monomials,
    differentiate_monomial,
    sort_monomial,
)


type CoefficientType = float | int
type PolynomialTermType = tuple[MonomialType, CoefficientType]
type PolynomialType = dict[MonomialType, CoefficientType]
type MaybePolynomialType = PolynomialType | None


def add_polynomial_terms_mutable(
    mutable: PolynomialType,
    terms: Iterable[PolynomialTermType],
) -> PolynomialType:
    for monomial, coefficient in terms:
        sorted_monomial = sort_monomial(monomial)

        if sorted_monomial in mutable:
            summation = mutable[sorted_monomial] + coefficient

            if math.isclose(summation, 0):
                del mutable[sorted_monomial]

            else:
                mutable[sorted_monomial] = summation

        else:
            mutable[sorted_monomial] = coefficient

    return mutable


def add_polynomial_terms_iterable(
    terms: Iterable[PolynomialTermType],
) -> MaybePolynomialType:
    return add_polynomial_terms_mutable(mutable={}, terms=terms)
    

def add_polynomial_iterable(
    polynomials: Iterable[PolynomialType],
) -> MaybePolynomialType:
    # filter_polynomials = iter(p for p in polynomials if p is not None)
    polynomials_iter = iter(polynomials)

    try:
        first = next(polynomials_iter)
    except StopIteration:
        return None

    # copy and sort monomial dictionary
    mutable = {
        sort_monomial(monomial): coefficient for monomial, coefficient in first.items()
    }

    assert len(mutable) == len(first)

    for polynomial in polynomials_iter:
        add_polynomial_terms_mutable(mutable=mutable, terms=polynomial.items())

    return mutable


def add_polynomials(left: PolynomialType, right: PolynomialType) -> MaybePolynomialType:
    if len(left) < len(right):
        left, right = right, left

    return add_polynomial_iterable((left, right))


def add_maybe_polynomials(
    left: MaybePolynomialType, right: MaybePolynomialType
) -> MaybePolynomialType:
    if left is None:
        return right

    if right is None:
        return left

    result = add_polynomials(left, right)

    if result:
        return result


def constant_polynomial(value: float) -> PolynomialType:
    return {tuple(): value}


def differentiate_polynomial(
    polynomial: PolynomialType, wrt: int
) -> MaybePolynomialType:
    """Differentiate a varaible with respect to a variable"""

    def gen_derivative_terms():
        for diff_monomial, coefficient in polynomial.items():
            result = differentiate_monomial(diff_monomial, wrt)

            # term has not disappeared
            if result:
                diff_monomial, power = result
                yield diff_monomial, coefficient * power

    # differentiated monomials are unique, hence we can use dict here
    result = dict(gen_derivative_terms())

    if result:
        return result


def gen_multiplication_terms(
    left: Iterable[PolynomialTermType], right: Iterable[PolynomialTermType]
) -> Iterable[PolynomialTermType]:
    for (left_monomial, left_coefficient), (
        right_monomial,
        right_coefficient,
    ) in itertools.product(left, right):
        coefficient = left_coefficient * right_coefficient

        if math.isclose(coefficient, 0, abs_tol=1e-12):
            continue

        monomial = add_monomials(left_monomial, right_monomial)

        yield monomial, coefficient


def is_zero(polynomial: PolynomialType):
    if len(polynomial) == 0:
        return True
    
    elif len(polynomial) == 1:
        if val := polynomial.get(tuple(), None):
            return math.isclose(val, 0)
        
    return False


def multiply_polynomials(
    left: PolynomialType, right: PolynomialType
) -> MaybePolynomialType:
    multiplication_terms = gen_multiplication_terms(left.items(), right.items())

    result = add_polynomial_terms_mutable(mutable={}, terms=multiplication_terms)

    # if empty dictionary, return None
    if result:
        return result


def multiply_polynomial_iterable(
    polynomials: Iterable[MaybePolynomialType],
) -> MaybePolynomialType:
    filter_polynomials = iter(d.items() for d in polynomials if d is not None)

    try:
        first = next(filter_polynomials)
    except StopIteration:
        return None

    *_, multiplication_terms = itertools.accumulate(
        filter_polynomials,
        gen_multiplication_terms,
        initial=first,
    )

    result = add_polynomial_terms_mutable(mutable={}, terms=multiplication_terms)

    # if empty dictionary, return None
    if result:
        return result


def multiply_with_scalar_mutable(
    mutable: PolynomialType,
    scalar: float,
):
    for monomial, coefficient in mutable.items():
        mutable[monomial] = coefficient * scalar

    return mutable


def multiply_with_scalar(
    polynomial: PolynomialType,
    scalar: float,
):
    def gen_terms():
        for monomial, coefficient in polynomial.items():
            yield monomial, coefficient * scalar

    return dict(gen_terms())
