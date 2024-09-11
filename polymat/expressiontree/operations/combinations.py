import abc

from itertools import combinations_with_replacement

from polymat.sparserepr.data.polynomial import (
    constant_polynomial,
    multiply_polynomial_iterable,
)
from polymat.utils.getstacklines import FrameSummaryMixin, to_operator_traceback
from polymat.expressiontree.nodes import SingleChildExpressionNode
from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.sparserepr.init import init_sparse_repr_from_iterable


class Combinations(FrameSummaryMixin, SingleChildExpressionNode):
    """
    Represents a combination polynomial operator that generates a polynomial vector 
    from a given input polynomial vector.

    The resulting polynomial vector contains all possible combinations of products of 
    the elements from the input polynomial vector, determined by the specified tuple of degrees. 

    Example:
        ```python
        x1 = polymat.define_variable('x1')
        x2 = polymat.define_variable('x2')
        x = polymat.v_stack((x1, x2))

        # Create combinations of the polynomial vector with degrees 0, 1, and 2
        expr = x.combinations(degrees=(0, 1, 2))

        # Convert the result to a sympy expression
        state, expr_sympy = polymat.to_sympy(expr.T).apply(state)

        # The output will be expr_sympy=Matrix([[1, x1, x2, x1**2, x1*x2, x2**2]])
        print(f'{expr_sympy=}')
        ```

    Args:
        degrees: A tuple representing the degrees of the elements in the output polynomial vector.

    Returns:
        A polynomial vector containing all the combinations of the input vector elements.
    """

    DegreeType = tuple[int, ...]

    def __str__(self):
        match self.degrees:
            case tuple():
                return f"combinations({self.child}, degrees={self.degrees})"
            case _:
                return f"combinations({self.child})"

    @property
    @abc.abstractmethod
    def degrees(self) -> DegreeType:
        """
        Degrees of the elements in the output polynomial vector
        """

    def apply(
        self,
        state: State,
    ) -> tuple[State, SparseRepr]:
        state, child = self.child.apply(state)

        if not (child.shape[1] == 1):
            raise AssertionError(
                to_operator_traceback(
                    message=f"{child.shape[1]=} is not 1",
                    stack=self.stack,
                )
            )

        def gen_combinations():
            for degree in self.degrees:
                yield from combinations_with_replacement(range(child.shape[0]), degree)

        combinations = tuple(gen_combinations())

        def gen_polynomial_matrix():
            for row, combination in enumerate(combinations):
                index = (row, 0)

                # x.combinations((0, 1, 2)) produces [1, x, x**2]
                if len(combination) == 0:
                    yield index, constant_polynomial(1.0)
                    continue

                polynomials = (child.at(row, 0) for row in combination)

                result = multiply_polynomial_iterable(polynomials)

                if result:
                    yield index, result

        return state, init_sparse_repr_from_iterable(
            data=gen_polynomial_matrix(),
            shape=(len(combinations), 1),
        )
