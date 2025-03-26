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
    expressions: Iterable[Iterable[MatrixExpression[State]]],
) -> MatrixExpression[State]: ...

class from_[State: BaseState]:
    def __new__(_, value: FromAnyTypes) -> MatrixExpression[State]: ...

class from_symmetric[State: BaseState]:
    def __new__(_, value: FromAnyTypes) -> SymmetricMatrixExpression[State]: ...

class from_vector[State: BaseState]:
    def __new__(_, value: FromAnyTypes) -> VectorExpression[State]: ...

class from_row_vector[State: BaseState]:
    def __new__(_, value: FromAnyTypes) -> RowVectorExpression[State]: ...

class from_polynomial[State: BaseState]:
    def __new__(_, value: FromAny.ValueType) -> ScalarPolynomialExpression[State]: ...

class define_variable[State: BaseState]:
    @overload
    def __new__(_, name: str) -> VariableExpression[State]: ...
    @overload
    def __new__(
        _, name: str, size: int | MatrixExpression[State] | None
    ) -> VariableVectorSymbolExpression[State]: ...

class from_variables[State: BaseState]:
    def __new__(_, value: FromVariables.VARIABLE_TYPE) -> VariableVectorExpression[State]: ...

class from_variable_indices[State: BaseState]:
    def __new__(_, indices: tuple[int, ...]) -> VariableVectorExpression[State]: ...

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
