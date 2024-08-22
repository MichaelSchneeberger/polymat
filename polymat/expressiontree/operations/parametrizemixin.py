from abc import abstractmethod
from typing_extensions import override

from polymat.expressiontree.operations.definevariablemixin import DefineVariableMixin
from polymat.sparserepr.init import init_sparse_repr_from_iterable
from polymat.sparserepr.sparsereprmixin import SparseReprMixin
from polymat.state import State
from polymat.utils.getstacklines import FrameSummaryMixin, to_operator_exception
from polymat.variable import Variable
from polymat.expressiontree.expressiontreemixin import SingleChildExpressionTreeMixin


class ParametrizeMixin(FrameSummaryMixin, SingleChildExpressionTreeMixin):
    """Caches the polynomial matrix using the state"""

    def __str__(self):
        return f"parametrize({self.child})"

    @property
    @abstractmethod
    def variable(self) -> Variable:
        """The symbol representing the variable."""

    @override
    def apply(self, state: State) -> tuple[State, SparseReprMixin]:
        state, child = self.child.apply(state)

        if not (child.shape[1] == 1):
            raise AssertionError(
                to_operator_exception(
                    message=f"{child.shape}",
                    stack=self.stack,
                )
            )

        nvar = child.shape[0]

        state, gen_polynomial_matrix = DefineVariableMixin.create_variable_vector(
            state,
            variable=self.variable,
            size=nvar,
            stack=self.stack,
        )

        return state, init_sparse_repr_from_iterable(
            data=gen_polynomial_matrix(), shape=(nvar, 1)
        )
