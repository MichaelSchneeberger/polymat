from abc import abstractmethod
from itertools import accumulate
import math

from typing_extensions import override

import numpy as np
import sympy
from sympy.polys.polyerrors import GeneratorsNeeded

from polymat.expressiontree.nodes import ExpressionNode
from polymat.sparserepr.data.monomial import sort_monomial
from polymat.sparserepr.data.polynomialmatrix import MatrixIndexType
from polymat.sparserepr.data.polynomial import PolynomialType, constant_polynomial
from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.sparserepr.init import init_from_polynomial_matrix
from polymat.utils.getstacklines import (
    FrameSummaryMixin,
    to_operator_traceback,
)
from polymat.symbol import Symbol


class FromAnyMixin(FrameSummaryMixin, ExpressionNode):
    VALUE_TYPES = float | int | np.number | sympy.Expr | ExpressionNode

    def __str__(self):
        if len(self.data) == 1:
            if len(self.data[0]) == 1:
                return str(self.data[0][0])

        def gen_elem():
            for row, row_data in enumerate(self.data):
                for col, elem in enumerate(row_data):
                    yield f"{(row, col)=}: {elem}"

        return "\n".join(gen_elem())

    @property
    @abstractmethod
    def data(self) -> tuple[tuple[VALUE_TYPES, ...], ...]:
        """The matrix of numbers in row major order."""

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        def acc_polynomial_matrix_data(
            acc: tuple[State, tuple[tuple[MatrixIndexType, PolynomialType], ...]],
            next: tuple[int, int, FromAnyMixin.VALUE_TYPES],
        ):
            state, data = acc
            row, col, entry = next

            matrix_index = (row, col)

            match entry:
                case ExpressionNode():
                    state, instance = entry.apply(state)

                    if not (instance.shape == (1, 1)):
                        raise AssertionError(
                            to_operator_traceback(
                                message=f"{instance.shape=} is not (1, 1)",
                                stack=self.stack,
                            )
                        )

                    entry = instance.at(0, 0)

                    if entry:
                        data = data + ((matrix_index, entry),)

                    return state, data

                case bool() | np.bool_():
                    value = int(entry)

                case np.number():
                    value = float(entry)

                case _:
                    value = entry

            match value:
                case int() | float():
                    if math.isclose(value, 0):
                        return state, data

                    polynomial = constant_polynomial(value)

                case sympy.Expr():
                    try:
                        sympy_poly = sympy.poly(value)

                    except GeneratorsNeeded:
                        if math.isclose(value, 0):
                            return state, data

                        polynomial = constant_polynomial(float(value))

                    except ValueError:
                        raise ValueError(f"{value=}")

                    else:
                        for symbol in sympy_poly.gens:
                            state, _ = state.register(
                                symbol=Symbol(str(symbol)),
                                size=1,
                                stack=self.stack,
                            )

                        def gen_polynomial():
                            # a5 x1 x3**2 -> c=a5, m_cnt=(1, 0, 2)
                            for value, variable_powers in zip(
                                sympy_poly.coeffs(), sympy_poly.monoms()
                            ):
                                if math.isclose(value, 0):
                                    continue

                                # m_cnt=(1, 0, 2) -> m=((0, 1) (1, 2))
                                def gen_monomial():
                                    for sympy_index, power in enumerate(variable_powers):
                                        if 0 < power:
                                            variable = Symbol(
                                                str(sympy_poly.gens[sympy_index])
                                            )
                                            index = state.indices[variable]
                                            yield (index.start, power)

                                monomial = sort_monomial(tuple(gen_monomial()))

                                yield monomial, value

                        polynomial = dict(gen_polynomial())

                case _:
                    raise AssertionError(
                        to_operator_traceback(
                            message=f"unknown data type {type(value)=}",
                            stack=self.stack,
                        )
                    )

            return state, data + ((matrix_index, polynomial),)

        def gen_entries():
            for row, row_data in enumerate(self.data):
                for col, entry in enumerate(row_data):
                    yield row, col, entry

        *_, (state, data) = accumulate(
            gen_entries(), acc_polynomial_matrix_data, initial=(state, tuple())
        )

        return state, init_from_polynomial_matrix(
            data=dict(data),
            shape=(len(self.data), len(self.data[0])),
        )
