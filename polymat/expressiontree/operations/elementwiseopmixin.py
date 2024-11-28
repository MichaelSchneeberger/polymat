from abc import abstractmethod

from typing_extensions import override

from polymat.expressiontree.nodes import TwoChildrenExpressionNode
from polymat.sparserepr.data.polynomial import MaybePolynomialType
from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.utils.getstacklines import FrameSummaryMixin, to_operator_traceback
from polymat.sparserepr.init import (
    init_sparse_repr_from_data,
    init_sparse_repr_from_iterable,
)


class ElementwiseOpMixin(FrameSummaryMixin, TwoChildrenExpressionNode):
    """
    Adds two polymatrices

        [[2*x1+x2], [x1**2]] + [[3*x2], [x1]]  ->  [[2*x1+4*x2], [x1+x1**2]].

    If one summand is of size (1, 1), then perform broadcast:

        [[2*x1+x2], [x1**2]] + [[x1]]  ->  [[3*x1+x2], [x1+x1**2]].
    """

    def __str__(self):
        return f"{self.operator_name}({self.left}, {self.right})"

    @staticmethod
    @abstractmethod
    def operator(
        left: MaybePolynomialType, right: MaybePolynomialType
    ) -> MaybePolynomialType: ...

    @property
    @abstractmethod
    def operator_name(self) -> str: ...

    @staticmethod
    @abstractmethod
    def is_addition() -> bool: ...

    @classmethod
    def scalar_operation(
        cls,
        left: SparseRepr,
        right: MaybePolynomialType,
    ) -> SparseRepr:
        match right:
            case None if cls.is_addition():
                return left
            
            case None:
                return init_sparse_repr_from_data({}, left.shape)
            
            case _:
                def gen_polynomial_matrix():
                    for index, left_polynomial in left.entries():
                        result = cls.operator(left_polynomial, right)

                        if result:
                            yield index, result

                return init_sparse_repr_from_iterable(
                    gen_polynomial_matrix(), left.shape
                )


    @override
    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, left = self.left.apply(state=state)
        state, right = self.right.apply(state=state)

        # def single_element_operation(left, right):
        #     left_polynomial = left.at(0, 0)

        #     if left_polynomial:

        #         def gen_polynomial_matrix():
        #             for index, right_polynomial in right.entries():
        #                 result = self.operator(right_polynomial, left_polynomial)

        #                 if result:
        #                     yield index, result

        #         return init_sparse_repr_from_iterable(
        #             gen_polynomial_matrix(), right.shape
        #         )

        #     elif self.is_addition:
        #         return right

        #     else:
        #         return init_sparse_repr_from_iterable(tuple(), right.shape)

        match (left.shape, right.shape):
            case ((1, 1), (1, 1)):
                left_polynomial = left.at(0, 0)
                right_polynomial = right.at(0, 0)

                result = self.operator(left_polynomial, right_polynomial)

                if result:
                    data = {(0, 0): result}
                else:
                    data = {}

                sparse_repr = init_sparse_repr_from_data(data, shape=left.shape)

            case ((1, 1), _):
                sparse_repr = self.scalar_operation(
                    left=right,
                    right=left.at(0, 0),
                )
                # sparse_repr = single_element_operation(left, right)

            case (_, (1, 1)):
                sparse_repr = self.scalar_operation(
                    left=left,
                    right=right.at(0, 0),
                )
                # sparse_repr = single_element_operation(right, left)

            case ((n_rows, n_cols), _):
                if left.shape != right.shape:
                    raise AssertionError(
                        to_operator_traceback(
                            message=(
                                f"Cannot do element-wise {self.operator_name} of matrices"
                                f"with shapes {left.shape} and {right.shape}."
                            ),
                            stack=self.stack,
                        )
                    )

                def gen_polynomial_matrix():
                    for row in range(n_rows):
                        for col in range(n_cols):
                            left_polynomial = left.at(row=row, col=col)
                            right_polynomial = right.at(row=row, col=col)

                            result = self.operator(left_polynomial, right_polynomial)

                            if result:
                                yield (row, col), result

                sparse_repr = init_sparse_repr_from_iterable(
                    gen_polynomial_matrix(), left.shape
                )

        return state, sparse_repr
