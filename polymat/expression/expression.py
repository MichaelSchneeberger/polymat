from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, override

from polymat.expressiontree.data.variables import VariableType
from polymat.expressiontree.from_ import (
    FromAnyTypes,
    from_any_or_raise_exception,
    from_any_or_none,
)
from polymat.expressiontree.nodes import (
    SingleChildExpressionNode,
    ExpressionNode,
)
from polymat.expressiontree.init import (
    init_addition,
    init_assert_polynomial,
    init_assert_vector,
    init_block_diagonal,
    init_cache,
    init_combinations,
    init_diagonal,
    init_differentiate,
    init_elementwise_mult,
    init_evaluate,
    init_filter_predicate,
    init_filter_non_zero,
    init_kronecker,
    init_linear_monomials,
    init_linear_coefficients,
    init_matrix_mult,
    init_product,
    init_quadratic_coefficients,
    init_quadratic_monomials,
    init_rep_mat,
    init_reshape,
    init_get_item,
    init_row_summation,
    init_to_symmetric_matrix,
    init_transpose,
    init_truncate_monomials,
    init_v_stack,
    init_variable_vector,
    from_vector_to_symmetric_matrix,
)
from polymat.expressiontree.operations.combinations import Combinations
from polymat.expressiontree.operations.filterpredicator import FilterPredicate
from polymat.expressiontree.operations.product import Product
from polymat.expressiontree.operations.truncatemonomials import TruncateMonomials
from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.utils.getstacklines import FrameSummary, get_frame_summary
from polymat.symbol import Symbol


class Expression(SingleChildExpressionNode, ABC):
    def __add__(self, other: FromAnyTypes):
        return self._binary(init_addition, self, other)

    def __getitem__(self, key):
        assert len(key) == 2

        return self.copy(
            child=init_get_item(
                child=self.child,
                key=key,
            )
        )

    def __matmul__(self, other: FromAnyTypes):
        return self._binary(init_matrix_mult, self, other)

    def __mul__(self, other: FromAnyTypes):
        return self._binary(init_elementwise_mult, self, other)

    def __neg__(self):
        return (-1) * self

    def __pow__(self, exponent: int):
        result = self
        for _ in range(exponent - 1):
            result = result * self
        return result

    def __radd__(self, other: FromAnyTypes):
        return self._binary(init_addition, other, self)

    def __rmul__(self, other: FromAnyTypes):
        return self._binary(init_elementwise_mult, other, self)

    def __rmatmul__(self, other: FromAnyTypes):
        return self._binary(init_matrix_mult, other, self)

    def __rsub__(self, other: FromAnyTypes):
        return other + (-self)

    def __str__(self):
        return str(self.child)

    def __sub__(self, other: FromAnyTypes):
        return self + (-other)

    def __truediv__(self, other: float | int):
        if not isinstance(other, float | int):
            return NotImplementedError

        return (1 / other) * self

    def _is_left_subtype_of_right(self, left, right):
        # overwrite this function when extending Expression
        return False

    def _binary(self, op, left, right) -> Expression:
        stack = get_frame_summary(index=4)

        if isinstance(left, Expression) and isinstance(right, Expression):
            child = op(left.child, right.child, stack)

            if self._is_left_subtype_of_right(left, right):
                return right.copy(child=child)
            else:
                return left.copy(child=child)

        elif isinstance(left, Expression):
            right = from_any_or_none(right, stack)

            if right is None:
                return NotImplemented

            return left.copy(child=op(left.child, right, stack))

        # else right is an Expression
        else:
            left = from_any_or_none(left, stack)

            if left is None:
                return NotImplemented

            return right.copy(child=op(left, right.child, stack))

    def _get_children(
        self, others: Iterable[Expression], stack: tuple[FrameSummary, ...]
    ) -> tuple[ExpressionNode, ...]:
        if isinstance(others, Expression):
            others = (others,)

        def gen_children():
            yield self.child

            for e in others:
                if isinstance(e, Expression):
                    expr = e.child
                else:
                    expr = from_any_or_raise_exception(e, stack=stack)

                yield expr

        # arrange blocks vertically and concatanete
        return tuple(gen_children())

    def _v_stack(
        self, others: Iterable[Expression], stack: tuple[FrameSummary, ...]
    ) -> ExpressionNode:
        # """ Vertically stack expressions """
        return init_v_stack(
            children=self._get_children(others, stack=stack),
            stack=get_frame_summary(),
        )

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        return self.child.apply(state)

    def assert_vector(self, stack=get_frame_summary()):
        return self.copy(child=init_assert_vector(stack=stack, child=self))

    def assert_polynomial(self, stack=get_frame_summary()):
        return self.copy(child=init_assert_polynomial(stack=stack, child=self))

    def block_diag(self, others: Iterable[Expression]):
        stack = get_frame_summary()

        return self.copy(
            child=init_block_diagonal(
                children=self._get_children(others, stack=stack),
            )
        )

    def cache(self):
        return self.copy(child=init_cache(self, stack=get_frame_summary()))

    # only applies to vector
    def combinations(self, degrees: Combinations.DegreeType):
        return self.copy(
            child=init_combinations(
                child=self.child,
                degrees=degrees,
                stack=get_frame_summary(),
            )
        )

    @abstractmethod
    def copy(self, /, **changes) -> Expression: ...

    def diag(self):
        return self.copy(
            child=init_diagonal(
                child=self.child,
                stack=get_frame_summary(),
            )
        )

    def diff(self, variables: VariableType):
        return self.copy(
            child=init_differentiate(
                child=self.child,
                variables=variables,
                stack=get_frame_summary(),
            )
        )
    
    SubstitutionType = dict[Symbol, tuple[float, ...]]

    def eval(self, substitutions: SubstitutionType):
        return self.copy(
            child=init_evaluate(
                child=self.child,
                substitutions=tuple(substitutions.items()),
                stack=get_frame_summary(),
            )
        )

    # only applies to vector
    def filter_predicate(self, predicate: FilterPredicate.PredicatorType):
        return self.copy(
            child=init_filter_predicate(
                child=self.child,
                predicate=predicate,
                stack=get_frame_summary(),
            )
        )

    def filter_non_zero(self):
        return self.copy(
            child=init_filter_non_zero(
                child=self.child,
                stack=get_frame_summary(),
            )
        )

    def h_stack(self, others: Iterable[Expression]):
        return self.T.v_stack((e.T for e in others)).T

    def kron(self, other: Expression):
        return self.copy(child=init_kronecker(left=self.child, right=other.child))

    # this method only applies to vectors
    def to_linear_coefficients(
        self,
        variables: VariableType,
        monomials: Expression | None = None,
    ):
        return self.copy(
            child=init_linear_coefficients(
                child=self.child,
                monomials=monomials,
                variables=variables,
                stack=get_frame_summary(),
            )
        )

    # deprecated method name, use to_linear_coefficient_vector instead
    def linear_in(
        self,
        variables: VariableType,
        monomials: Expression | None = None,
    ):
        return self.to_linear_coefficients(variables=variables, monomials=monomials)

    def to_linear_monomials(self, variables: Expression):
        return self.copy(
            child=init_linear_monomials(
                child=self.child,
                variables=variables,
            )
        )

    # deprecated method name, use to_linear_monomials instead
    def linear_monomials_in(self, variables: Expression):
        return self.to_linear_monomials(variables=variables)

    def product(
        self,
        others: Iterable[Expression],
        degrees: Product.DegreeType = None,
    ):
        stack = get_frame_summary()

        return self.copy(
            child=init_product(
                children=self._get_children(others, stack=stack),
                stack=stack,
                degrees=degrees,
            )
        )

    # this method only applies to polynomials
    def to_gram_matrix(
        self,
        variables: Expression,
        monomials: Expression | None = None,
    ):
        return self.copy(
            child=init_to_symmetric_matrix(
                child=init_quadratic_coefficients(
                    child=self.child,
                    monomials=monomials,
                    variables=variables,
                    stack=get_frame_summary(),
                )
            )
        )

    def to_quadratic_coefficients(
        self,
        variables: Expression,
        monomials: Expression | None = None,
    ):
        return self.to_gram_matrix(variables=variables, monomials=monomials)

    # deprecated method name, use to_quadratic_coefficient_matrix instead
    def quadratic_in(
        self,
        variables: Expression,
        monomials: Expression | None = None,
    ):
        return self.to_gram_matrix(variables=variables, monomials=monomials)

    def to_quadratic_monomials(self, variables: VariableType):
        return self.copy(
            child=init_quadratic_monomials(
                child=self.child,
                variables=variables,
            )
        )

    # deprecated method name, use to_quadratic_monomials instead
    def quadratic_monomials_in(self, variables: VariableType):
        return self.to_quadratic_monomials(variables=variables)

    def rep_mat(self, n: int, m: int):
        return self.copy(
            child=init_rep_mat(
                child=self.child,
                repetition=(n, m),
            ),
        )

    def reshape(self, n: int, m: int):
        return self.copy(
            child=init_reshape(
                child=self.child,
                new_shape=(n, m),
            )
        )

    def sum(self):
        """
        sum all elements of each row
        """

        return self.copy(
            child=init_row_summation(
                child=self.child,
            )
        )

    def symmetric(self):
        return self.copy(child=init_to_symmetric_matrix(child=self.child))

    @property
    def T(self):
        return self.copy(child=init_transpose(self.child))

    def to_symmetric_matrix(self):
        return self.copy(
            child=from_vector_to_symmetric_matrix(
                child=self.child,
                stack=get_frame_summary(),
            )
        )

    def to_monomial_vector(self):
        return self.assert_vector(stack=get_frame_summary())

    def to_polynomial(self):
        return self.assert_polynomial(stack=get_frame_summary())[0, 0]

    def to_variable_vector(self):
        return self.copy(
            child=init_variable_vector(
                child=self.child,
            )
        )

    def to_vector(self):
        return self.assert_vector(stack=get_frame_summary())

    def trace(self):
        return self.diag().T.sum()

    def truncate_monomials(
        self, variables: VariableType, degeees: TruncateMonomials.DegreeType
    ):
        return self.copy(
            child=init_truncate_monomials(
                child=self.child, variables=variables, degrees=degeees
            ),
        )

    def v_stack(self, others: Iterable[Expression]):
        stack = get_frame_summary()
        return self.copy(child=self._v_stack(others=others, stack=stack))


class VariableExpression(Expression):
    @property
    @abstractmethod
    def symbol(self) -> Symbol: ...

    def iterate_symbols(self):
        yield self.symbol
