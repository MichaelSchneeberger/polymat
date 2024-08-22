from typing import Iterable

from polymat.utils.getstacklines import get_stack_lines
from polymat.variable import Variable
from polymat.utils import typing
from polymat.expression.abc import Expression, VectorExpression
from polymat.expressiontree.operations.fromvariablesmixin import FromVariablesMixin
from polymat.expressiontree.init import (
    init_define_variable,
    init_from_,
    init_from_variables,
    init_from_variable_indices,
)
from polymat.expression.init import (
    init_expression,
    init_variable_expression,
)


def _split_first[T: Expression](
    expressions: Iterable[T],
) -> tuple[T, tuple[T, ...]]:
    expressions_iter = iter(expressions)

    first = next(expressions_iter)

    if not isinstance(first, Expression):
        first = init_expression(from_(first))

    return first, tuple(expressions_iter)


def block_diag(expressions: Iterable[Expression]) -> Expression:
    first, others = _split_first(expressions)
    return first.block_diag(others=others)


def concat(expressions: Iterable[Iterable[Expression]]):
    def gen_h_stack():
        for col_expressions in expressions:
            yield h_stack(col_expressions)

    return v_stack(gen_h_stack())


def from_(value: typing.FROM_TYPES):
    stack = get_stack_lines()
    return init_expression(init_from_(value, stack=stack))


# use for type hinting
from_symmetric = from_
from_vector = from_
from_row_vector = from_
from_polynomial = from_


def define_variable(
    name: str | Variable,
    size: int | Expression | None = None,
):
    if not isinstance(name, Variable):
        variable = Variable(name)
    else:
        variable = name

    if isinstance(size, Expression):
        n_size = size.child
    else:
        n_size = size

    return init_variable_expression(
        child=init_define_variable(
            variable=variable, size=n_size, stack=get_stack_lines()
        ),
        variable=variable,
    )


def from_variables(variables: FromVariablesMixin.VARIABLE_TYPE):
    return init_expression(init_from_variables(variables=variables))


def from_variable_indices(indices: tuple[int, ...]):
    return init_expression(init_from_variable_indices(indices=indices))


def h_stack(expressions: Iterable[Expression]) -> Expression:
    return v_stack((expr.T for expr in expressions)).T


def product(expressions: Iterable[VectorExpression]) -> VectorExpression:
    first, others = _split_first(expressions)
    return first.product(others=others)


def v_stack(expressions: Iterable[Expression]) -> Expression:
    first, others = _split_first(expressions)
    return first.v_stack(others=others)
