from abc import abstractmethod
import itertools
from typing import Iterable, override

from polymat.sparserepr.data.monomial import PowerVariableType, MonomialType
from polymat.sparserepr.data.polynomialmatrix import MatrixIndexType
from polymat.sparserepr.data.polynomial import (
    CoefficientType,
    PolynomialType,
    add_polynomial_terms_iterable,
)
from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.expressiontree.nodes import SingleChildExpressionNode
from polymat.sparserepr.init import init_sparse_repr_from_iterable
from polymat.utils.getstacklines import (
    FrameSummaryMixin,
    to_operator_traceback,
)
from polymat.symbol import Symbol


class Evaluate(FrameSummaryMixin, SingleChildExpressionNode):
    SubstitutionType = tuple[tuple[Symbol, tuple[float, ...]], ...]

    @property
    @abstractmethod
    def substitutions(self) -> SubstitutionType: ...

    def __str__(self):
        return f"eval({self.child}, {self.substitutions})"

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, child = self.child.apply(state=state)

        def acc_indices_and_values(
            acc: tuple[State, dict[int, float]], next: tuple[Symbol, tuple[float, ...]]
        ):
            state, mapping = acc
            symbol, values = next

            index_range = state.get_index_range(symbol)

            if index_range is None:
                return acc

            if len(values) == 1:
                values = tuple(values[0] for _ in index_range)

            else:
                if not (len(index_range) == len(values)):
                    raise AssertionError(
                        to_operator_traceback(
                            message=(
                                f"Cannot replace symbol {symbol} that has an index range {index_range} with {values}, "
                                f"because the {len(index_range)} does not equal {len(values)}."
                            ),
                            stack=self.stack,
                        )
                    )

            for index, value in zip(index_range, values):
                mapping[index] = value

            return state, mapping

        *_, (state, index_value_map) = itertools.accumulate(
            self.substitutions,
            acc_indices_and_values,
            initial=(state, {}),
        )

        def gen_polynomial_matrix() -> Iterable[tuple[MatrixIndexType, PolynomialType]]:
            for index, polynomial in child.entries():

                def gen_polynomial_data() -> (
                    Iterable[tuple[MonomialType, CoefficientType]]
                ):
                    for monomial, coeff in polynomial.items():

                        def acc_monomial(
                            acc: tuple[MonomialType, CoefficientType],
                            power_var: PowerVariableType,
                        ):
                            monomial, coeff = acc
                            index, power = power_var

                            if index in index_value_map:
                                new_coeff = coeff * index_value_map[index] ** power
                                return monomial, new_coeff

                            else:
                                return monomial + (power_var,), coeff

                        *_, (eval_monomial, eval_coeff) = tuple(
                            itertools.accumulate(
                                monomial,
                                acc_monomial,
                                initial=(tuple(), coeff),
                            )
                        )

                        yield eval_monomial, eval_coeff

                result = add_polynomial_terms_iterable(terms=gen_polynomial_data())

                if result:
                    yield index, result

        return state, init_sparse_repr_from_iterable(
            data=gen_polynomial_matrix(),
            shape=child.shape,
        )
