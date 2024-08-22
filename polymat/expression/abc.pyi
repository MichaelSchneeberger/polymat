from typing import Iterable, overload

from polymat.expressiontree.operations.filtermixin import FilterMixin
from polymat.expressiontree.expressiontreemixin import (
    ExpressionTreeMixin,
    SingleChildExpressionTreeMixin,
)
from polymat.expressiontree.operations.getitemmixin import GetItemMixin
from polymat.sparserepr.sparsereprmixin import SparseReprMixin
from polymat.state import State
from polymat.utils.typing import FROM_TYPES
from polymat.variable import Variable

class Expression(SingleChildExpressionTreeMixin):
    def __add__(self, other: FROM_TYPES) -> Expression: ...
    def __getitem__(self, key: GetItemMixin.KEY_TYPE) -> PolynomialExpression: ...
    @overload
    def __matmul__(self, other: VectorExpression) -> VectorExpression: ...
    @overload
    def __matmul__(self, other: FROM_TYPES) -> Expression: ...
    def __mul__(self, other: FROM_TYPES) -> Expression: ...
    def __neg__(self) -> Expression: ...
    def __pow__(self, exponent: int) -> Expression: ...
    def __radd__(self, other: FROM_TYPES) -> Expression: ...
    def __rmul__(self, other: FROM_TYPES) -> Expression: ...
    def __rmatmul__(self, other: FROM_TYPES) -> Expression: ...
    def __rsub__(self, other: FROM_TYPES) -> Expression: ...
    def __sub__(self, other: FROM_TYPES) -> Expression: ...
    def __truediv__(self, other: float | int) -> Expression: ...
    def apply(self, state: State) -> tuple[State, SparseReprMixin]: ...
    def block_diag(self, others: Iterable[Expression]) -> Expression: ...
    def cache(self) -> Expression: ...
    def copy(self, child: ExpressionTreeMixin) -> Expression: ...
    def diff(self, variables: VariableVectorExpression) -> Expression: ...
    def eval(self, substitutions: dict[Variable, tuple[float, ...]]) -> Expression: ...
    def h_stack(self, others: Iterable[Expression]) -> Expression: ...
    def kron(self, other: Expression) -> Expression: ...
    def linear_monomials_in(
        self, variables: VariableVectorExpression
    ) -> MonomialVectorExpression: ...
    def quadratic_monomials_in(
        self, variables: VariableVectorExpression
    ) -> MonomialVectorExpression: ...
    def rep_mat(self, n: int, m: int) -> Expression: ...
    def reshape(self, n: int, m: int) -> Expression: ...
    def slice(self, slice: tuple[tuple[int, ...], tuple[int, ...]]) -> Expression: ...
    def sum(self) -> VectorExpression: ...
    def symmetric(self) -> SymmetricExpression: ...
    def to_polynomial(self) -> PolynomialExpression: ...
    def to_variable_vector(self) -> VariableVectorExpression: ...
    def to_vector(self) -> VectorExpression: ...
    def v_stack(self, others: Iterable[Expression]) -> Expression: ...
    @property
    def T(self) -> Expression: ...

class SymmetricExpression(Expression):
    @overload
    def __add__(self, other: SymmetricExpression) -> SymmetricExpression: ...
    @overload
    def __add__(self, other: FROM_TYPES) -> Expression: ...
    @overload
    def __matmul__(self, other: VectorExpression) -> VectorExpression: ...
    @overload
    def __matmul__(self, other: SymmetricExpression) -> SymmetricExpression: ...
    @overload
    def __matmul__(self, other: FROM_TYPES) -> Expression: ...
    @overload
    def __mul__(self, other: SymmetricExpression) -> SymmetricExpression: ...
    @overload
    def __mul__(self, other: FROM_TYPES) -> Expression: ...
    def __neg__(self) -> SymmetricExpression: ...
    def __pow__(self, exponent: int) -> SymmetricExpression: ...
    def __radd__(self, other: FROM_TYPES) -> SymmetricExpression: ...
    def __rmul__(self, other: FROM_TYPES) -> SymmetricExpression: ...
    def __rsub__(self, other: FROM_TYPES) -> SymmetricExpression: ...
    @overload
    def __sub__(self, other: SymmetricExpression) -> SymmetricExpression: ...
    @overload
    def __sub__(self, other: FROM_TYPES) -> Expression: ...
    @overload
    def block_diag(
        self, others: Iterable[SymmetricExpression]
    ) -> SymmetricExpression: ...
    @overload
    def block_diag(self, others: Iterable[Expression]) -> Expression: ...
    def cache(self) -> SymmetricExpression: ...
    def copy(self, child: ExpressionTreeMixin) -> SymmetricExpression: ...
    def diag(self) -> VectorExpression: ...
    def diff(self, variables: VariableVectorExpression) -> SymmetricExpression: ...
    def eval(
        self, substitutions: dict[Variable, tuple[float, ...]]
    ) -> SymmetricExpression: ...
    @overload
    def kron(self, other: SymmetricExpression) -> SymmetricExpression: ...
    @overload
    def kron(self, other: Expression) -> Expression: ...
    @property
    def T(self) -> SymmetricExpression: ...
    def trace(self) -> PolynomialExpression: ...

class VectorExpression(Expression):
    def __add__(self, other: FROM_TYPES) -> VectorExpression: ...
    def __mul__(self, other: FROM_TYPES) -> VectorExpression: ...
    def __neg__(self) -> VectorExpression: ...
    def __pow__(self, exponent: int) -> VectorExpression: ...
    def __radd__(self, other: FROM_TYPES) -> VectorExpression: ...
    @overload
    def __matmul__(self, other: RowVectorExpression) -> SymmetricExpression: ...
    @overload
    def __matmul__(self, other: FROM_TYPES) -> Expression: ...
    def __rmul__(self, other: FROM_TYPES) -> VectorExpression: ...
    def __rsub__(self, other: FROM_TYPES) -> VectorExpression: ...
    def __sub__(self, other: FROM_TYPES) -> VectorExpression: ...
    def __truediv__(self, other: float | int) -> VectorExpression: ...
    def cache(self) -> VectorExpression: ...
    def combinations(self, degrees: tuple[int, ...]) -> VectorExpression: ...
    def copy(self, child: ExpressionTreeMixin) -> VectorExpression: ...
    def diag(self) -> SymmetricExpression: ...
    def diff(self, variables: VariableVectorExpression) -> VectorExpression: ...
    def eval(
        self, substitutions: dict[Variable, tuple[float, ...]]
    ) -> VectorExpression: ...
    def filter(self, predicator: FilterMixin.PREDICATOR_TYPE) -> VectorExpression: ...
    @overload
    def kron(self, other: VectorExpression) -> VectorExpression: ...
    @overload
    def kron(self, other: Expression) -> Expression: ...
    def linear_in(
        self,
        variables: VariableVectorExpression,
        monomials: MonomialVectorExpression | None = None,
    ) -> Expression: ...
    def parametrize(self, variable: Variable | str) -> MultiDimVariableExpression: ...
    def product(self, others: Iterable[VectorExpression]) -> VectorExpression: ...
    def sum(self) -> PolynomialExpression: ...
    def v_stack(self, others: Iterable[VectorExpression]) -> VectorExpression: ...
    @property
    def T(self) -> RowVectorExpression: ...

class RowVectorExpression(Expression):
    def __add__(self, other: FROM_TYPES) -> RowVectorExpression: ...
    def __mul__(self, other: FROM_TYPES) -> RowVectorExpression: ...
    def __neg__(self) -> RowVectorExpression: ...
    def __pow__(self, exponent: int) -> RowVectorExpression: ...
    def __radd__(self, other: FROM_TYPES) -> RowVectorExpression: ...
    @overload
    def __matmul__(self, other: SymmetricExpression) -> RowVectorExpression: ...
    @overload
    def __matmul__(self, other: VectorExpression) -> PolynomialExpression: ...
    @overload
    def __matmul__(self, other: FROM_TYPES) -> Expression: ...
    def __rmul__(self, other: FROM_TYPES) -> RowVectorExpression: ...
    def __rsub__(self, other: FROM_TYPES) -> RowVectorExpression: ...
    def __sub__(self, other: FROM_TYPES) -> RowVectorExpression: ...
    def __truediv__(self, other: float | int) -> RowVectorExpression: ...
    def cache(self) -> RowVectorExpression: ...
    def copy(self, child: ExpressionTreeMixin) -> RowVectorExpression: ...
    def diff(self, variables: VariableVectorExpression) -> RowVectorExpression: ...
    def eval(
        self, substitutions: dict[Variable, tuple[float, ...]]
    ) -> RowVectorExpression: ...
    def h_stack(self, others: Iterable[RowVectorExpression]) -> RowVectorExpression: ...
    @overload
    def kron(self, other: VectorExpression) -> RowVectorExpression: ...
    @overload
    def kron(self, other: Expression) -> Expression: ...
    def sum(self) -> PolynomialExpression: ...
    @property
    def T(self) -> VectorExpression: ...

class PolynomialExpression(VectorExpression):
    def __add__(self, other: FROM_TYPES) -> PolynomialExpression: ...
    def __mul__(self, other: FROM_TYPES) -> PolynomialExpression: ...
    def __neg__(self) -> PolynomialExpression: ...
    def __pow__(self, exponent: int) -> PolynomialExpression: ...
    def __radd__(self, other: FROM_TYPES) -> PolynomialExpression: ...
    def __rmul__(self, other: FROM_TYPES) -> PolynomialExpression: ...
    def __rsub__(self, other: FROM_TYPES) -> PolynomialExpression: ...
    def __sub__(self, other: FROM_TYPES) -> PolynomialExpression: ...
    def __truediv__(self, other: float | int) -> PolynomialExpression: ...
    def cache(self) -> PolynomialExpression: ...
    def copy(self, child: ExpressionTreeMixin) -> PolynomialExpression: ...
    def diff(self, variables: VariableVectorExpression) -> RowVectorExpression: ...
    def eval(
        self, substitutions: dict[Variable, tuple[float, ...]]
    ) -> PolynomialExpression: ...
    def h_stack(self, others: Iterable[Expression]) -> RowVectorExpression: ...
    def quadratic_in(
        self,
        variables: VariableVectorExpression,
        monomials: MonomialVectorExpression | None = None,
    ) -> SymmetricExpression: ...
    def sum(self) -> PolynomialExpression: ...

class MonomialVectorExpression(VectorExpression):
    @overload
    def __mul__(self, other: MonomialVectorExpression) -> MonomialVectorExpression: ...
    @overload
    def __mul__(self, other: FROM_TYPES) -> VectorExpression: ...
    def __pow__(self, exponent: int) -> MonomialVectorExpression: ...
    def cache(self) -> MonomialVectorExpression: ...
    def copy(self, child: ExpressionTreeMixin) -> MonomialVectorExpression: ...
    def filter(
        self, predicator: FilterMixin.PREDICATOR_TYPE
    ) -> MonomialVectorExpression: ...
    @overload
    def product(
        self, others: Iterable[MonomialVectorExpression]
    ) -> MonomialVectorExpression: ...
    @overload
    def product(self, others: Iterable[VectorExpression]) -> VectorExpression: ...

class MonomialExpression(PolynomialExpression, MonomialVectorExpression):
    @overload
    def __mul__(self, other: MonomialExpression) -> MonomialExpression: ...
    @overload
    def __mul__(self, other: FROM_TYPES) -> PolynomialExpression: ...
    def __pow__(self, exponent: int) -> MonomialExpression: ...
    def cache(self) -> MonomialExpression: ...
    def copy(self, child: ExpressionTreeMixin) -> MonomialExpression: ...

class VariableVectorExpression(MonomialVectorExpression):
    def __pow__(self, exponent: int) -> MonomialVectorExpression: ...
    def cache(self) -> VariableVectorExpression: ...
    def combinations(self, degrees: tuple[int, ...]) -> MonomialVectorExpression: ...
    def copy(self, child: ExpressionTreeMixin) -> VariableVectorExpression: ...
    def filter(
        self, predicator: FilterMixin.PREDICATOR_TYPE
    ) -> VariableVectorExpression: ...

class VariableExpression(VariableVectorExpression):
    def cache(self) -> VariableExpression: ...
    def copy(self, child: ExpressionTreeMixin) -> VariableExpression: ...
    @property
    def variable(self) -> Variable: ...

class SingleDimVariableExpression(MonomialExpression, VariableExpression):
    def cache(self) -> SingleDimVariableExpression: ...
    def copy(self, child: ExpressionTreeMixin) -> SingleDimVariableExpression: ...
    @property
    def variable(self) -> Variable: ...

# class ParametrizedPolynomialExpression(PolynomialExpression):
#     @property
#     def variable(self) -> Variable: ...
