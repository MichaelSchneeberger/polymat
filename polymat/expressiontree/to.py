import itertools
import math
import sympy

import numpy as np
from numpy.typing import NDArray

from dataclassabc import dataclassabc

import statemonad
from statemonad.abc import StateMonadNode
from statemonad.typing import StateMonad

from polymat.utils.getstacklines import get_stack_lines
from polymat.variable import Variable
from polymat.arrayrepr.abc import ArrayRepr
from polymat.arrayrepr.init import init_array_repr
from polymat.sparserepr.data.monomial import (
    MonomialType,
    monomial_degree,
    monomial_degree_in,
)
from polymat.sparserepr.sparsereprmixin import SparseReprMixin
from polymat.state import State
from polymat.expressiontree.expressiontreemixin import ExpressionTreeMixin
from polymat.expressiontree.init import init_assert_vector


def to_array(
    expr: ExpressionTreeMixin,
    variables: ExpressionTreeMixin,
) -> StateMonad[State, ArrayRepr]:
    """
    Given a monomial of degree d, this function returns the indices of a monomial
    vector containing monomials of the same degree.

    Given the variable mapping {x : 0, y : 1}, the monomial x*y of the monomial vector

        z = [x**2, x*y, x*y, y**2]

    results in indices = (1, 2).

    Or, the monomial x*y**2 of the monomial vector

        z = [x**3, x**2*y, x**2*y, x*y**2, x**2*y, x*y**2, x*y**2, y**3]

    results in indices = (3, 5, 6)
    """

    @dataclassabc(frozen=True)
    class ToArrayStateMonadTree(StateMonadNode):
        expr: ExpressionTreeMixin
        variables: ExpressionTreeMixin

        def __str__(self):
            return f"to_array({self.expr}, {self.variables})"

        def apply(self, state: State):
            state, polymatrix = init_assert_vector(
                child=expr,
                stack=get_stack_lines(),
            ).apply(state)
            state, variables = self.variables.apply(state)

            indices = tuple(variables.to_indices())
            n_param = len(indices)
            index_to_linear_column = {index: col for col, index in enumerate(indices)}

            assert len(index_to_linear_column) == len(indices)

            array_repr = init_array_repr(
                n_eq=polymatrix.shape[0],
                n_param=n_param,
            )

            for row in range(polymatrix.shape[0]):
                polynomial = polymatrix.at(row, 0)

                if polynomial is None:
                    continue

                for monomial, value in polynomial.items():

                    def gen_linear_columns():
                        for index, power in monomial:
                            if index not in index_to_linear_column:
                                name = state.get_name(index)
                                names = set(
                                    state.get_name(index)
                                    for index in index_to_linear_column.keys()
                                )
                                raise Exception(
                                    f'Variable "{name}" not contained in {names}'
                                )

                            linear_col = index_to_linear_column[index]

                            for _ in range(power):
                                yield linear_col

                    linear_columns = tuple(gen_linear_columns())

                    # NP: document this function, especially magic return line
                    def to_column_indices(
                        n_var: int,
                        variable_indices: tuple[int, ...],
                    ) -> set[int]:
                        variable_indices_perm = itertools.permutations(variable_indices)

                        return set(
                            sum(
                                idx * (n_var**level)
                                for level, idx in enumerate(monomial)
                            )
                            for monomial in variable_indices_perm
                        )

                    columns = to_column_indices(n_param, linear_columns)

                    # print(f'{columns=}')

                    col_value = value / len(columns)

                    for col in columns:
                        array_repr.add(row, col, monomial_degree(monomial), col_value)

            return state, array_repr

    return statemonad.from_node(ToArrayStateMonadTree(expr=expr, variables=variables))


def to_degree(
    expr: ExpressionTreeMixin,
    variables: ExpressionTreeMixin | None = None,
) -> StateMonad[State, NDArray]:
    @dataclassabc(frozen=True)
    class ToDegreeStateMonadTree(StateMonadNode):
        expr: ExpressionTreeMixin
        variables: ExpressionTreeMixin | None = None

        def __str__(self):
            return f"to_degree({self.expr}, {self.variables})"

        def apply(self, state: State):
            state, polymatrix = self.expr.apply(state)

            A = np.zeros(polymatrix.shape, dtype=np.double)

            if self.variables:
                state, variables_ = to_variable_indices(self.variables).apply(state)

                def get_degree(monomial: MonomialType):
                    return monomial_degree_in(monomial, set(variables_))

            else:

                def get_degree(monomial: MonomialType):
                    return monomial_degree(monomial)

            for (row, col), polynomial in polymatrix.entries():

                def gen_degrees():
                    for monomial in polynomial.keys():
                        yield get_degree(monomial)

                A[row, col] = max(gen_degrees())

            return state, A

    return statemonad.from_node(ToDegreeStateMonadTree(expr=expr, variables=variables))


def to_numpy(
    expr: ExpressionTreeMixin, assert_constant: bool = True
) -> StateMonad[State, NDArray]:
    @dataclassabc(frozen=True)
    class ToNumpyStateMonadTree(StateMonadNode):
        expr: ExpressionTreeMixin
        assert_constant: bool

        def __str__(self):
            return f"to_numpy({self.expr})"

        def apply(self, state: State):
            state, polymatrix = self.expr.apply(state)

            A = np.zeros(polymatrix.shape, dtype=np.double)

            for (row, col), polynomial in polymatrix.entries():
                for monomial, value in polynomial.items():
                    if len(monomial) == 0:
                        A[row, col] = value

                    elif assert_constant:
                        raise Exception(f"non-constant term {monomial=}")

            return state, A

    return statemonad.from_node(
        ToNumpyStateMonadTree(expr=expr, assert_constant=assert_constant)
    )


def to_shape(expr: ExpressionTreeMixin) -> StateMonad[State, tuple[int, int]]:
    @dataclassabc(frozen=True)
    class ToShapeStateMonadTree(StateMonadNode):
        expr: ExpressionTreeMixin

        def __str__(self):
            return f"to_shape({self.expr})"

        def apply(self, state: State):
            state, polymatrix = expr.apply(state)

            return state, polymatrix.shape

    return statemonad.from_node(ToShapeStateMonadTree(expr=expr))


def to_sparse_repr(expr: ExpressionTreeMixin) -> StateMonad[State, SparseReprMixin]:
    return statemonad.from_node(expr)


def to_sympy(expr: ExpressionTreeMixin) -> StateMonad[State, sympy.Expr]:
    @dataclassabc(frozen=True)
    class ToSympyStateMonadTree(StateMonadNode):
        expr: ExpressionTreeMixin

        def __str__(self):
            return f"to_sympy({self.expr})"

        def apply(self, state: State):
            state, polymatrix = expr.apply(state)

            sympy_matrix = sympy.zeros(*polymatrix.shape)

            for (row, col), polynomial in polymatrix.entries():
                # print(f'{row=}, {col=}, {polynomial=}')
                sympy_poly_terms = []
                for monomial, coeff in polynomial.items():
                    sympy_monomial = math.prod(
                        sympy.Symbol(state.get_name(index)) ** power
                        for index, power in monomial
                    )

                    if math.isclose(coeff, 1.0):
                        # no need to add 1 in front
                        sympy_poly_terms.append(sympy_monomial)

                    else:
                        sympy_poly_terms.append(coeff * sympy_monomial)

                sympy_matrix[row, col] = sum(sympy_poly_terms)

            if math.prod(polymatrix.shape) == 1:
                # just return the expression
                sympy_matrix = sympy_matrix[0, 0]

            return state, sympy_matrix

    return statemonad.from_node(ToSympyStateMonadTree(expr=expr))


def to_tuple(
    expr: ExpressionTreeMixin, assert_constant: bool = True
) -> StateMonad[State, tuple[tuple[float, ...], ...]]:
    @dataclassabc(frozen=True)
    class ToTupleStateMonadTree(StateMonadNode):
        expr: ExpressionTreeMixin
        assert_constant: bool

        def __str__(self):
            return f"to_tuple({self.expr})"

        def apply(self, state: State):
            state, polymatrix = self.expr.apply(state)

            n_rows, n_cols = polymatrix.shape

            def gen_tuple():
                for row in range(n_rows):

                    def gen_column():
                        for col in range(n_cols):
                            polynomial = polymatrix.at(row, col)

                            if polynomial is None:
                                yield 0

                            else:
                                # find constant monomial
                                for monomial, value in polynomial.items():
                                    if len(monomial) == 0:
                                        yield value

                                    elif assert_constant:
                                        raise Exception(
                                            f"non-constant term {monomial=}"
                                        )

                    yield tuple(gen_column())

            return state, tuple(gen_tuple())

    return statemonad.from_node(
        ToTupleStateMonadTree(expr=expr, assert_constant=assert_constant)
    )


def to_variable_indices(
    expr: ExpressionTreeMixin,
) -> StateMonad[State, tuple[int, ...]]:
    @dataclassabc(frozen=True)
    class ToVariableIndicesStateMonadTree(StateMonadNode):
        expr: ExpressionTreeMixin

        def __str__(self):
            return f"to_variable_indices({self.expr})"

        def apply(self, state: State):
            state, polymatrix = self.expr.apply(state)

            # keep order of variables indices
            indices = tuple(polymatrix.to_indices())

            return state, indices

    return statemonad.from_node(ToVariableIndicesStateMonadTree(expr=expr))


def to_variables(
    expr: ExpressionTreeMixin,
) -> StateMonad[State, tuple[Variable, ...]]:
    @dataclassabc(frozen=True)
    class ToVariablesStateMonadTree(StateMonadNode[State, tuple[Variable, ...]]):
        expr: ExpressionTreeMixin

        def __str__(self):
            return f"to_variables({self.expr})"

        def apply(self, state: State):
            state, polymatrix = self.expr.apply(state)

            unsorted_variables = (
                state.get_variable(index) for index in polymatrix.to_indices()
            )

            # no need to sort variables
            variables = tuple(set(unsorted_variables))

            return state, variables

    return statemonad.from_node(ToVariablesStateMonadTree(expr=expr))
