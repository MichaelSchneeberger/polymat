from typing import Iterable, overload

from polymat.state.state import State as BaseState
from polymat.expressiontree.sources.fromany import FromAny
from polymat.expressiontree.sources.fromvariables import FromVariables
from polymat.expressiontree.operations.product import Product
from polymat.expressiontree.from_ import FromAnyTypes
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

@overload
def block_diag[State: BaseState](
    expressions: Iterable[SymmetricMatrixExpression[State]],
) -> SymmetricMatrixExpression[State]: ...
@overload
def block_diag[State: BaseState](
    expressions: Iterable[MatrixExpression[State]],
) -> MatrixExpression[State]: ...
def concat[State: BaseState](
    expressions: Iterable[Iterable[MatrixExpression]],
) -> MatrixExpression[State]: ...
def from_[State: BaseState](value: FromAnyTypes) -> MatrixExpression[State]: ...
def from_symmetric[State: BaseState](
    value: FromAnyTypes,
) -> SymmetricMatrixExpression[State]: ...
def from_vector[State: BaseState](value: FromAnyTypes) -> VectorExpression[State]: ...
def from_row_vector[State: BaseState](
    value: FromAnyTypes,
) -> RowVectorExpression[State]: ...
def from_polynomial[State: BaseState](
    value: FromAny.ValueType,
) -> ScalarPolynomialExpression[State]: ...
@overload
def define_variable[State: BaseState](
    name: str,
    size: int | MatrixExpression[State] | None,
) -> VariableVectorSymbolExpression[State]: ...
@overload
def define_variable[State: BaseState](
    name: str,
) -> VariableExpression[State]: ...
def from_variables[State: BaseState](
    variables: FromVariables.VARIABLE_TYPE,
) -> VariableVectorExpression[State]: ...
def from_variable_indices[State: BaseState](
    indices: tuple[int, ...],
) -> VariableVectorExpression[State]: ...
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
