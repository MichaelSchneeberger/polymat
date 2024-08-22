import sympy
from numpy.typing import NDArray
from polymat.expressiontree.operations.fromanymixin import FromAnyMixin


# Types that can be converted to an Expression
FROM_TYPES = (
    FromAnyMixin.ELEM_TYPES
    | NDArray
    | sympy.Matrix
    | tuple[FromAnyMixin.VALUE_TYPES]
    | tuple[tuple[FromAnyMixin.VALUE_TYPES]]
)
