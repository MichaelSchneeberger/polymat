from typing import Iterable

from polymat.symbols.strsymbol import StrSymbol
from polymat.symbols.symbol import Symbol
from polymat.utils.getstacklines import get_frame_summary
from polymat.expressiontree.from_ import FromAnyTypes, from_any_or_raise_exception
from polymat.expressiontree.sources.fromvariables import FromVariables
from polymat.expressiontree.operations.definevariable import DefineVariable
from polymat.expressiontree.operations.product import Product
from polymat.expressiontree.init import (
    init_define_variable,
    init_from_variables,
    init_from_variable_indices,
)
from polymat.expression.typedexpressions import (
    MatrixExpression,
    VectorExpression,
)
from polymat.expression.init import (
    init_expression,
    init_variable_expression,
)


def _split_first[T: FromAnyTypes | MatrixExpression](
    expressions: Iterable[T],
) -> tuple[T, tuple[T, ...]]:
    expressions_iter = iter(expressions)

    # raises exception if iterable is empty
    first = next(expressions_iter)

    if not isinstance(first, MatrixExpression):
        first = init_expression(from_(first))

    others = tuple(expressions_iter)

    return first, others  # type: ignore


def block_diag(
    expressions: Iterable[MatrixExpression],
) -> MatrixExpression:
    first, others = _split_first(expressions)
    return first.block_diag(others=others)


def concat(expressions: Iterable[tuple[MatrixExpression]]):
    def gen_rows_expr():
        for row in expressions:
            if len(row) == 1:
                yield row[0]
            else:
                yield h_stack(row)

    rows_expr = tuple(gen_rows_expr())

    if len(rows_expr) == 1:
        return rows_expr[0]
    else:
        return v_stack(rows_expr)


def define_variable(
    name: str | Symbol,
    size: DefineVariable.SizeType | None = None,
):
    if not isinstance(name, Symbol):
        symbol = StrSymbol(name)
    else:
        symbol = name

    if isinstance(size, MatrixExpression):
        n_size = size.child
    else:
        n_size = size

    return init_variable_expression(
        child=init_define_variable(
            symbol=symbol, size=n_size, stack=get_frame_summary()
        ),
        symbol=symbol,
    )


def from_(value: FromAnyTypes | MatrixExpression):
    stack = get_frame_summary()
    return init_expression(from_any_or_raise_exception(value, stack=stack))


# used for type hinting
from_symmetric = from_
from_vector = from_
from_row_vector = from_
from_polynomial = from_


def from_variables(variables: FromVariables.VARIABLE_TYPE):
    return init_expression(init_from_variables(variables=variables))


def from_variable_indices(indices: tuple[int, ...]):
    return init_expression(
        init_from_variable_indices(
            indices=indices,
        )
    )


def h_stack(expressions: Iterable[MatrixExpression]) -> MatrixExpression:
    return v_stack((expr.T for expr in expressions)).T


def product(
    expressions: Iterable[VectorExpression],
    degrees: Product.DegreeType = None,
) -> VectorExpression:
    first, others = _split_first(expressions)
    return first.product(others=others, degrees=degrees)


def v_stack(expressions: Iterable[MatrixExpression]) -> MatrixExpression:
    first, others = _split_first(expressions)
    return first.v_stack(others=others)
