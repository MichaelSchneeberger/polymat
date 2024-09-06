import abc
from typing import override

from polymat.expressiontree.nodes import (
    ExpressionNode,
    SingleChildExpressionNode,
)
from polymat.sparserepr.data.monomial import split_monomial_indices
from polymat.sparserepr.init import init_sparse_repr_from_iterable
from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.utils.getstacklines import FrameSummaryMixin, to_operator_traceback


class QuadraticCoefficients(FrameSummaryMixin, SingleChildExpressionNode):
    """
    Constructs the coefficient matrix Q corresponding to the quadratic form of the polynomial.
    Given a vector of variables and, optionally, a vector of monomials, this class generates the coefficient
    matrix involved in the quadratic form p(x) = Z(x)^\top Q Z(x), where Q is the coefficient matrix 
    and Z(x) is the vector of monomials. If the vector of monomials is not provided, then it wil be
    computed using the to_quadratic_monomials method.

    Example:
        ```python
        x1 = polymat.define_variable('x1')
        x2 = polymat.define_variable('x2')
        x = polymat.v_stack((x1, x2))

        # Example polynomial
        p = 1 + x1*x2 + x1**2

        # Compute the coefficient matrix of the quadratic form
        expr = p.to_gram_matrix(x)

        # Convert the result to a sympy expression
        state, expr_sympy = polymat.to_sympy(expr.T).apply(state)

        # The output will be 
        # expr_sympy=Matrix([
        # [1,   0,   0],     
        # [0,   1, 0.5],     
        # [0, 0.5,   0]])  
        print(f'{expr_sympy=}')

        Args:
            x: The vector of variables
            monomials: A vector of monomials. If not provided, it will be computed.

        Returns:
            An instance representing the vector of monomials involved in the quadratic form of the polynomial.
        ```
    """
    
    @property
    @abc.abstractmethod
    def monomials(self) -> ExpressionNode: ...

    @property
    @abc.abstractmethod
    def variables(self) -> ExpressionNode: ...

    @property
    @abc.abstractmethod
    def ignore_unmatched(self) -> bool: ...

    def __str__(self):
        return f"quadratic_in({self.child}, {self.variables})"

    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, child = self.child.apply(state=state)
        state, monomial_vector = self.monomials.apply(state=state)
        state, variable_vector = self.variables.apply(state=state)

        if not (child.shape[1] == 1):
            raise AssertionError(
                to_operator_traceback(
                    message=f"{child.shape[1]=} is not 1",
                    stack=self.stack,
                )
            )

        # keep order of monomials
        monomials = tuple(monomial_vector.to_monomials())

        indices = set(variable_vector.to_indices())

        for row in range(child.shape[0]):
            polynomial = child.at(row, 0)

            if polynomial is None:
                continue

            def gen_polymatrix():
                for monomial, value in polynomial.items():  # type: ignore
                    x_monomial = tuple(
                        (index, count) for index, count in monomial if index in indices
                    )
                    p_monomial = tuple(
                        (index, count)
                        for index, count in monomial
                        if index not in indices
                    )

                    left, right = split_monomial_indices(x_monomial)

                    try:
                        col = monomials.index(left)
                    except ValueError:
                        raise AssertionError(
                            to_operator_traceback(
                                message=f"{left=} not in {monomials}",
                                stack=self.stack,
                            )
                        )

                    try:
                        row = monomials.index(right)
                    except ValueError:
                        raise AssertionError(
                            to_operator_traceback(
                                message=f"{right=} not in {monomials}",
                                stack=self.stack,
                            )
                        )

                    yield (row, col), {p_monomial: value}

        polymatrix = init_sparse_repr_from_iterable(
            data=gen_polymatrix(),
            shape=(monomial_vector.shape[0], monomial_vector.shape[0]),
        )

        return state, polymatrix
