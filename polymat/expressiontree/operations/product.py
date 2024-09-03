import abc

from itertools import product

from polymat.sparserepr.data.polynomial import multiply_polynomial_iterable
from polymat.utils.getstacklines import FrameSummaryMixin, to_operator_traceback
from polymat.expressiontree.nodes import MultiChildrenExpressionNode
from polymat.sparserepr.sparserepr import SparseRepr
from polymat.state import State
from polymat.sparserepr.init import init_from_polynomial_matrix


class Product(FrameSummaryMixin, MultiChildrenExpressionNode):
    # FIXME: improve docstring
    """
    combination using degrees=(0, 1, 2):

    [[x], [y]]  ->  [[1], [x], [y], [x**2], [x*y], [y**2]]
    """

    DEGREE_TYPES = tuple[int, ...] | None

    def __str__(self):
        children = ",".join(str(c) for c in self.children)

        match self.degrees:
            case tuple():
                return f"product({children}, degrees={self.degrees})"
            case _:
                return f"product({children})"

    @property
    @abc.abstractmethod
    def degrees(self) -> DEGREE_TYPES:
        """
        Vector or scalar expression, or a list of integers.
        """

    def apply(self, state: State) -> tuple[State, SparseRepr]:
        state, children = self.apply_children(state)

        for child in children:
            if not (child.shape[1] == 1):
                raise AssertionError(
                    to_operator_traceback(
                        message=f"{child.shape[1]=} is not 1",
                        stack=self.stack,
                    )
                )

        product_rows = product(*(range(e.shape[0]) for e in children))

        if self.degrees is not None:
            degrees = self.degrees

            product_rows = filter(lambda v: sum(v) in degrees, product_rows)

        def gen_polynomial_matrix():
            for output_row, sel_product_rows in enumerate(product_rows):
                # select one polynomial from each child
                polynomials = (
                    polymatrix.at(row, 0)
                    for polymatrix, row in zip(children, sel_product_rows)
                )

                result = multiply_polynomial_iterable(polynomials)

                if result:
                    yield (output_row, 0), result

        data = dict(gen_polynomial_matrix())

        return state, init_from_polynomial_matrix(
            data=data,
            shape=(len(data), 1),
        )
