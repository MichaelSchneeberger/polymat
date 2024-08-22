from typing import Callable
import numpy as np
import sympy

from dataclassabc import dataclassabc
from numpy.typing import NDArray

from polymat.expressiontree.operations.assertshapemixin import AssertShapeMixin
from polymat.expressiontree.operations.blockdiagonalmixin import (
    BlockDiagonalMixin,
)
from polymat.expressiontree.operations.cachemixin import CacheMixin
from polymat.expressiontree.operations.diagmixin import DiagMixin
from polymat.expressiontree.operations.evalmixin import EvalMixin
from polymat.expressiontree.operations.filtermixin import FilterMixin
from polymat.expressiontree.operations.fromanymixin import FromAnyMixin
from polymat.expressiontree.operations.fromsparsereprmixin import FromSparseReprMixin
from polymat.expressiontree.operations.fromvariableindicesmixin import (
    FromVariableIndicesMixin,
)
from polymat.expressiontree.operations.fromvariablesmixin import FromVariablesMixin
from polymat.expressiontree.operations.kronmixin import KronMixin
from polymat.expressiontree.operations.linearinmixin import LinearInExprMixin
from polymat.expressiontree.operations.linearmonomialsmixin import (
    LinearMonomialsMixin,
)
from polymat.expressiontree.operations.parametrizemixin import ParametrizeMixin
from polymat.expressiontree.operations.productmixin import ProductMixin
from polymat.expressiontree.operations.quadraticinmixin import (
    QuadraticInExprMixin,
)
from polymat.expressiontree.operations.quadraticmonomialsmixin import (
    QuadraticMonomialsMixin,
)
from polymat.expressiontree.operations.repmatmixin import RepMatMixin
from polymat.expressiontree.operations.reshapemixin import ReshapeMixin
from polymat.expressiontree.operations.getitemmixin import GetItemMixin
from polymat.expressiontree.operations.summixin import SumMixin
from polymat.expressiontree.operations.symmetricmixin import SymmetricMixin
from polymat.expressiontree.operations.tovariablevectormixin import (
    ToVariableVectorMixin,
)
from polymat.expressiontree.expressiontreemixin import ExpressionTreeMixin
from polymat.expressiontree.operations.additionmixin import AdditionMixin
from polymat.expressiontree.operations.combinationsmixin import (
    CombinationsMixin,
)
from polymat.expressiontree.operations.differentiatemixin import (
    DifferentiateMixin,
)
from polymat.expressiontree.operations.elementwisemultmixin import (
    ElementwiseMultMixin,
)
from polymat.expressiontree.operations.fromnumpymixin import FromNumpyMixin
from polymat.expressiontree.operations.definevariablemixin import (
    DefineVariableMixin,
)
from polymat.expressiontree.operations.matrixmultmixin import MatrixMultMixin
from polymat.expressiontree.operations.transposemixin import TransposeMixin
from polymat.expressiontree.operations.vstackmixin import VStackMixin
from polymat.sparserepr.sparsereprmixin import SparseReprMixin
from polymat.variable import Variable
from polymat.utils.getstacklines import FrameSummary
from polymat.utils import typing


@dataclassabc(frozen=True, repr=False)
class AdditionExprImpl(AdditionMixin):
    left: ExpressionTreeMixin
    right: ExpressionTreeMixin
    stack: tuple[FrameSummary]


def init_addition(
    left: ExpressionTreeMixin,
    right: ExpressionTreeMixin,
    stack: tuple[FrameSummary],
):
    return AdditionExprImpl(left=left, right=right, stack=stack)


@dataclassabc(frozen=True, repr=False)
class AssertShapeImpl(AssertShapeMixin):
    child: ExpressionTreeMixin
    fn: Callable[[int, int], bool]
    msg: Callable[[int, int], str]
    stack: tuple[FrameSummary]

    def __repr__(self):
        return repr(self.child)


init_assert_shape = AssertShapeImpl


def init_assert_vector(child: ExpressionTreeMixin, stack: tuple[FrameSummary]):
    return init_assert_shape(
        child=child,
        stack=stack,
        fn=lambda row, col: col == 1,
        msg=lambda row, col: f"number of column {col} must be 1",
    )


def init_assert_polynomial(child: ExpressionTreeMixin, stack: tuple[FrameSummary]):
    return init_assert_shape(
        child=child,
        stack=stack,
        fn=lambda row, col: row == 1 and col == 1,
        msg=lambda row, col: f"number of row {row} and column {col} must be both 1",
    )


@dataclassabc(frozen=True)
class BlockDiagonalImpl(BlockDiagonalMixin):
    children: tuple[ExpressionTreeMixin, ...]


init_block_diagonal = BlockDiagonalImpl


@dataclassabc(frozen=True, repr=False)
class CacheImpl(CacheMixin):
    child: ExpressionTreeMixin
    stack: tuple[FrameSummary, ...]


init_cache = CacheImpl


@dataclassabc(frozen=True, repr=False)
class CombinationsImpl(CombinationsMixin):
    child: ExpressionTreeMixin
    degrees: tuple[int, ...]
    stack: tuple[FrameSummary]


def init_combinations(
    child: ExpressionTreeMixin,
    degrees: tuple[int, ...],
    stack: tuple[FrameSummary],
):
    assert len(degrees)

    return CombinationsImpl(
        child=child,
        degrees=degrees,
        stack=stack,
    )


@dataclassabc(frozen=True, repr=False)
class DifferentiateImpl(DifferentiateMixin):
    child: ExpressionTreeMixin
    variables: ExpressionTreeMixin
    stack: tuple[FrameSummary]


init_differentiate = DifferentiateImpl


@dataclassabc(frozen=True, repr=False)
class DiagImpl(DiagMixin):
    child: ExpressionTreeMixin
    stack: tuple[FrameSummary]


init_diag = DiagImpl


@dataclassabc(frozen=True, repr=False)
class ElementwiseMultImpl(ElementwiseMultMixin):
    left: ExpressionTreeMixin
    right: ExpressionTreeMixin
    stack: tuple[FrameSummary]


init_elementwise_mult = ElementwiseMultImpl


@dataclassabc(frozen=True, repr=False)
class EvalImpl(EvalMixin):
    child: ExpressionTreeMixin
    substitutions: EvalMixin.SUBSTITUTION_TYPE
    stack: tuple[FrameSummary]


def init_eval(
    child: ExpressionTreeMixin,
    substitutions: dict[Variable, tuple[float, ...]],
    stack: tuple[FrameSummary],
):
    return EvalImpl(
        child=child,
        substitutions=tuple(substitutions.items()),
        stack=stack,
    )


@dataclassabc(frozen=True, repr=False)
class FilterImpl(FilterMixin):
    child: ExpressionTreeMixin
    predicator: FilterMixin.PREDICATOR_TYPE
    stack: tuple[FrameSummary]


# default constructor
init_filter = FilterImpl


@dataclassabc(frozen=True)
class FromNumpyImpl(FromNumpyMixin):
    data: NDArray


init_from_numpy = FromNumpyImpl


@dataclassabc(frozen=True, repr=False)
class FromAnyImpl(FromAnyMixin):
    data: tuple[tuple[FromAnyMixin.ELEM_TYPES]]
    stack: tuple[FrameSummary, ...]


def init_from_any(
    data: tuple[tuple[FromAnyMixin.ELEM_TYPES]],
    stack: tuple[FrameSummary, ...],
):
    return FromAnyImpl(
        data=data,
        stack=stack,
    )


@dataclassabc(frozen=True)
class FromSparseReprImpl(FromSparseReprMixin):
    sparse_repr: SparseReprMixin


def init_from_sparse_repr(sparse_repr: SparseReprMixin):
    return FromSparseReprImpl(sparse_repr=sparse_repr)


@dataclassabc(frozen=True, repr=False)
class DefineVariableImpl(DefineVariableMixin):
    variable: Variable
    size: int | ExpressionTreeMixin
    stack: tuple[FrameSummary, ...]


def init_define_variable(
    variable: Variable,
    stack: tuple[FrameSummary, ...],
    size: int | ExpressionTreeMixin | None = None,
):
    if size is None:
        size = 1

    return DefineVariableImpl(variable=variable, size=size, stack=stack)


@dataclassabc(frozen=True)
class FromVariablesImpl(FromVariablesMixin):
    variables: FromVariablesMixin.VARIABLE_TYPE


init_from_variables = FromVariablesImpl


@dataclassabc(frozen=True)
class FromVariableIndicesImpl(FromVariableIndicesMixin):
    indices: tuple[int, ...]


init_from_variable_indices = FromVariableIndicesImpl


def init_from_or_none(
    value: typing.FROM_TYPES, stack: tuple[FrameSummary, ...]
) -> ExpressionTreeMixin | None:
    """
    Create an expression object from a value, or give value_if_not_supported if
    the expression cannot be constructed from the given value.
    """
    if isinstance(value, int | float | np.number):
        wrapped = ((value,),)
        return init_from_any(wrapped, stack=stack)

    elif isinstance(value, np.ndarray):
        # Case when it is a (n,) array
        if len(value.shape) != 2:
            value = value.reshape(-1, 1)

        # if value.dtype == np.object_ or True:

        def gen_elements():
            for row in value:
                if isinstance(row, np.ndarray):
                    yield tuple(row)
                else:
                    yield (row,)

        return init_from_any(tuple(gen_elements()), stack=stack)
        # else:
        #     return init_from_numpy(value)

    elif isinstance(value, sympy.Matrix):
        data = tuple(tuple(v for v in value.row(row)) for row in range(value.rows))
        return init_from_any(data, stack)

    elif isinstance(value, sympy.Expr):
        data = ((sympy.expand(value),),)
        return init_from_any(data, stack)

    elif isinstance(value, tuple):
        if isinstance(value[0], tuple):
            n_col = len(value[0])
            assert all(len(col) == n_col for col in value)

            data = value

        else:
            data = tuple((e,) for e in value)

        return init_from_any(data, stack)

    elif isinstance(value, ExpressionTreeMixin):
        return value


def init_from_(value: typing.FROM_TYPES, stack: tuple[FrameSummary, ...]):
    """
    Attempt create an expression object from a value. Raises an exception if
    the expression cannot be constructed from given value.
    """
    if v := init_from_or_none(value, stack):
        return v

    raise ValueError(
        "Unsupported type. Cannot construct expression "
        f"from value {value} with type {type(value)}"
    )


@dataclassabc(frozen=True)
class KronImpl(KronMixin):
    left: ExpressionTreeMixin
    right: ExpressionTreeMixin


init_kron = KronImpl


@dataclassabc(frozen=True, repr=False)
class LinearInExprImpl(LinearInExprMixin):
    child: ExpressionTreeMixin
    monomials: ExpressionTreeMixin
    variables: ExpressionTreeMixin
    ignore_unmatched: bool
    stack: tuple[FrameSummary]


def init_linear_in(
    child: ExpressionTreeMixin,
    variables: ExpressionTreeMixin,
    stack: tuple[FrameSummary],
    monomials: ExpressionTreeMixin | None = None,
    ignore_unmatched: bool = False,
):
    if monomials is None:
        monomials = init_linear_monomials(
            child=child,
            variables=variables,
        )

    return LinearInExprImpl(
        child=child,
        variables=variables,
        monomials=monomials,
        ignore_unmatched=ignore_unmatched,
        stack=stack,
    )


@dataclassabc(frozen=True)
class LinearMonomialsImpl(LinearMonomialsMixin):
    child: ExpressionTreeMixin
    variables: ExpressionTreeMixin


init_linear_monomials = LinearMonomialsImpl


@dataclassabc(frozen=True, repr=False)
class MatrixMultImpl(MatrixMultMixin):
    left: ExpressionTreeMixin
    right: ExpressionTreeMixin
    stack: tuple[FrameSummary]


init_matrix_mult = MatrixMultImpl


@dataclassabc(frozen=True, repr=False)
class ParametrizeImpl(ParametrizeMixin):
    child: ExpressionTreeMixin
    variable: Variable
    stack: tuple[FrameSummary]


def init_parametrize(
    child: ExpressionTreeMixin,
    variable: Variable | str,
    stack: tuple[FrameSummary],
):
    if not isinstance(variable, Variable):
        variable = Variable(variable)

    return ParametrizeImpl(
        child=child,
        variable=variable,
        stack=stack,
    )


@dataclassabc(frozen=True, repr=False)
class ProductImpl(ProductMixin):
    children: tuple[ExpressionTreeMixin, ...]
    degrees: ProductMixin.DEGREE_TYPES
    stack: tuple[FrameSummary]


def init_product(
    children: tuple[ExpressionTreeMixin, ...],
    stack: tuple[FrameSummary],
    degrees: ProductMixin.DEGREE_TYPES,
):
    return ProductImpl(
        children=children,
        stack=stack,
        degrees=degrees,
    )


@dataclassabc(frozen=True, repr=False)
class QuadraticInExprImpl(QuadraticInExprMixin):
    child: ExpressionTreeMixin
    monomials: ExpressionTreeMixin
    variables: ExpressionTreeMixin
    ignore_unmatched: bool
    stack: tuple[FrameSummary]


def init_quadratic_in(
    child: ExpressionTreeMixin,
    variables: ExpressionTreeMixin,
    stack: tuple[FrameSummary],
    monomials: ExpressionTreeMixin | None = None,
    ignore_unmatched: bool = False,
):
    if monomials is None:
        monomials = init_quadratic_monomials(child=child, variables=variables)

    return QuadraticInExprImpl(
        child=child,
        variables=variables,
        monomials=monomials,
        ignore_unmatched=ignore_unmatched,
        stack=stack,
    )


@dataclassabc(frozen=True)
class QuadraticMonomialsImpl(QuadraticMonomialsMixin):
    child: ExpressionTreeMixin
    variables: ExpressionTreeMixin


init_quadratic_monomials = QuadraticMonomialsImpl


@dataclassabc(frozen=True)
class GetItemImpl(GetItemMixin):
    child: ExpressionTreeMixin
    key: GetItemMixin.KEY_TYPE


init_get_item = GetItemImpl


@dataclassabc(frozen=True)
class SumImpl(SumMixin):
    child: ExpressionTreeMixin


def init_sum(child: ExpressionTreeMixin):
    return SumImpl(child=child)


@dataclassabc(frozen=True)
class SymmetricImpl(SymmetricMixin):
    child: ExpressionTreeMixin


init_symmetric = SymmetricImpl


@dataclassabc(frozen=True)
class RepMatImpl(RepMatMixin):
    child: ExpressionTreeMixin
    repetition: tuple[int, int]


init_rep_mat = RepMatImpl


@dataclassabc(frozen=True)
class ReshapeImpl(ReshapeMixin):
    child: ExpressionTreeMixin
    new_shape: tuple[int, int]


init_reshape = ReshapeImpl


@dataclassabc(frozen=True)
class ToVariableVectorImpl(ToVariableVectorMixin):
    child: ExpressionTreeMixin


init_variable_vector = ToVariableVectorImpl


@dataclassabc(frozen=True)
class TransposeImpl(TransposeMixin):
    child: ExpressionTreeMixin


init_transpose = TransposeImpl


@dataclassabc(frozen=True, repr=False)
class VStackImpl(VStackMixin):
    children: tuple[ExpressionTreeMixin, ...]
    stack: tuple[FrameSummary]


init_v_stack = VStackImpl
