import numpy as np
from numpy.typing import NDArray
import sympy

from polymat.expressiontree.init import init_from_any
from polymat.expressiontree.nodes import ExpressionNode
from polymat.expressiontree.operations.fromany import FromAny
from polymat.utils.getstacklines import FrameSummary


# Types that can be converted to an Expression
FromAnyTypes = (
    FromAny.ValueType
    | NDArray
    | sympy.Matrix
    | tuple[FromAny.ValueType, ...]
    | tuple[tuple[FromAny.ValueType, ...], ...]
)


def from_any_or_none(
    value: FromAnyTypes, stack: tuple[FrameSummary, ...]
) -> ExpressionNode | None:
    """
    Create an expression object from a value, or give value_if_not_supported if
    the expression cannot be constructed from the given value.
    """
    if isinstance(value, int | float | np.number):
        wrapped = ((value,),)
        return init_from_any(wrapped, stack=stack)

    elif isinstance(value, np.ndarray):
        # Case when it is a (n,) array
        if len(value.shape) != 2:
            value = value.reshape(-1, 1)

        # if value.dtype == np.object_ or True:

        def gen_elements():
            for row in value:
                if isinstance(row, np.ndarray):
                    yield tuple(row)
                else:
                    yield (row,)

        return init_from_any(tuple(gen_elements()), stack=stack)
        # else:
        #     return init_from_numpy(value)

    elif isinstance(value, sympy.Matrix):
        data = tuple(tuple(v for v in value.row(row)) for row in range(value.rows))
        return init_from_any(data, stack)

    elif isinstance(value, sympy.Expr):
        data = ((sympy.expand(value),),)
        return init_from_any(data, stack)

    elif isinstance(value, tuple):
        if isinstance(value[0], tuple):
            n_col = len(value[0])
            assert all(len(col) == n_col for col in value)

            data = value

        else:
            data = tuple((e,) for e in value)

        return init_from_any(data, stack)

    elif isinstance(value, ExpressionNode):
        return value


def from_any_or_raise_exception(
    value: FromAnyTypes, stack: tuple[FrameSummary, ...]
):
    """
    Attempt create an expression object from a value. Raises an exception if
    the expression cannot be constructed from given value.
    """
    if v := from_any_or_none(value, stack):
        return v

    raise ValueError(
        "Unsupported type. Cannot construct expression "
        f"from value {value} with type {type(value)}"
    )
