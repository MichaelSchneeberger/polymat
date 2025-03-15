from typing import Iterable, overload

from polymat.state.state import State as BaseState
from polymat.expression.typedexpressions import (
    MatrixExpression,
    RowVectorExpression,
    SymmetricMatrixExpression,
    VectorExpression,
    ScalarPolynomialExpression,
    VariableExpression,
    VariableVectorExpression,
    VariableVectorSymbolExpression,
)
from polymat.expressiontree.from_ import FromAnyTypes
from polymat.expressiontree.operations.fromany import FromAny
from polymat.expressiontree.operations.fromvariables import FromVariables
from polymat.expressiontree.operations.product import Product

@overload
def block_diag(
    expressions: Iterable[SymmetricMatrixExpression],
) -> SymmetricMatrixExpression: ...
@overload
def block_diag(expressions: Iterable[MatrixExpression]) -> MatrixExpression: ...
def concat(expressions: Iterable[Iterable[MatrixExpression]]): ...
def from_(value: FromAnyTypes) -> MatrixExpression: ...
def from_symmetric(value: FromAnyTypes) -> SymmetricMatrixExpression: ...
def from_vector(value: FromAnyTypes) -> VectorExpression: ...
def from_row_vector(value: FromAnyTypes) -> RowVectorExpression: ...
def from_polynomial(value: FromAny.ValueType) -> ScalarPolynomialExpression: ...
@overload
def define_variable(
    name: str,
    size: int | MatrixExpression | None,
) -> VariableVectorSymbolExpression[BaseState]: ...
@overload
def define_variable(
    name: str,
) -> VariableExpression[BaseState]: ...
def from_variables(
    variables: FromVariables.VARIABLE_TYPE,
) -> VariableVectorExpression[BaseState]: ...
def from_variable_indices(
    indices: tuple[int, ...],
) -> VariableVectorExpression[BaseState]: ...
@overload
def h_stack[State: BaseState](
    expressions: Iterable[RowVectorExpression[State]],
) -> RowVectorExpression[State]: ...
@overload
def h_stack[State: BaseState](
    expressions: Iterable[MatrixExpression[State]],
) -> MatrixExpression[State]: ...
def product[State: BaseState](
    expressions: Iterable[VectorExpression[State]], degrees: Product.DegreeType = None
) -> VectorExpression[State]: ...
@overload
def v_stack[State: BaseState](
    expressions: Iterable[VariableVectorSymbolExpression[State]],
) -> VariableVectorExpression[State]: ...
@overload
def v_stack[State: BaseState](
    expressions: Iterable[VariableVectorExpression[State]],
) -> VariableVectorExpression[State]: ...
@overload
def v_stack[State: BaseState](
    expressions: Iterable[VectorExpression[State]],
) -> VectorExpression[State]: ...
@overload
def v_stack[State: BaseState](
    expressions: Iterable[MatrixExpression[State]],
) -> MatrixExpression[State]: ...
