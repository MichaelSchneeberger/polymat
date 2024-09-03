import sympy
from numpy.typing import NDArray
from polymat.expressiontree.operations.fromany import FromAny


# Types that can be converted to an Expression
FROM_TYPES = (
    FromAny.VALUE_TYPES
    | NDArray
    | sympy.Matrix
    | tuple[FromAny.VALUE_TYPES, ...]
    | tuple[tuple[FromAny.VALUE_TYPES, ...], ...]
)
