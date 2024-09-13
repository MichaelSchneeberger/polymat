from typing import Iterable

type IndexType = int
type PowerType = int
type PowerVariableType = tuple[IndexType, PowerType]
type MonomialType = tuple[PowerVariableType, ...]

type MutableMonomialType = dict[IndexType, PowerType]


def add_monomials(
    left: MonomialType,
    right: MonomialType,
) -> MonomialType:
    if len(left) < len(right):
        left, right = right, left

    # copy monomial
    monomial = dict(left)

    # add element of right to mutable dictionary monomial
    add_monomials_mutable(monomial, right)

    return tuple(monomial.items())


def add_monomials_mutable(
    mutable: MutableMonomialType,
    other: MonomialType,
) -> MutableMonomialType:
    """
    (x1**2 x2, x2**2)  ->  x1**2 x2**3

    or in terms of indices {x1: 0, x2: 1}:

        (
            ((0, 2), (1, 1)),      # x1**2 x2
            ((1, 2),)              # x2**2
        )  ->  ((0, 2), (1, 3))    # x1**1 x2**3
    """

    for index, count in other:
        if index in mutable:
            mutable[index] += count
        else:
            mutable[index] = count

    return mutable


def differentiate_monomial(
    monomial: MonomialType, index: int
) -> tuple[MonomialType, int] | None:
    diff_monomial = []
    power = None

    for m_index, m_power in monomial:
        if m_index == index:
            power = m_power

            if 1 < m_power:
                diff_monomial.append((m_index, m_power - 1))

        else:
            diff_monomial.append((m_index, m_power))

    # monomial does not contain variable
    if power is None:
        return None

    return tuple(diff_monomial), power


def monomial_degree(monomial: MonomialType) -> int:
    """Degree of the monomial"""
    if not monomial:
        return 0

    return sum(power for _, power in monomial)


def monomial_degree_in(monomial: MonomialType, variables: set[int]) -> int:
    """Degree of the monomial"""
    if not monomial:
        return 0

    return sum(power for index, power in monomial if index in variables)


def sort_monomial(monomial: MonomialType) -> MonomialType:
    assert len(monomial) == len({index for index, _ in monomial})

    return tuple(sorted(monomial, key=lambda m: m[0]))


# NP: Sort list of monomials according to ... what?
def sort_monomials(monomials: Iterable[MonomialType]) -> tuple[MonomialType, ...]:
    """
    Sort list of monomials according to:

        1. degree of the monomial
        2. the number of variables appearing in the monomial
    """

    def key(monomial: MonomialType):
        return (monomial_degree(monomial), len(monomial))

    return tuple(sorted(monomials, key=key))


# NP: what does this function do? split according to what?
def split_monomial_indices(monomial: MonomialType) -> tuple[MonomialType, MonomialType]:
    left = []
    right = []

    is_left = True

    for index, power in monomial:
        count_left = power // 2

        if power % 2:
            if is_left:
                count_left = count_left + 1

            is_left = not is_left

        count_right = power - count_left

        if 0 < count_left:
            left.append((index, count_left))

        if 0 < count_right:
            right.append((index, power - count_left))

    return tuple(left), tuple(right)
