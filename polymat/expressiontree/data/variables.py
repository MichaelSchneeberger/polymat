from polymat.expressiontree.nodes import ExpressionNode
from polymat.state import State


type VariableType = ExpressionNode | tuple[int, ...]


def to_indices(state: State, variables: VariableType):
    match variables:
        case ExpressionNode():
            n_state, variable_vector = variables.apply(state=state)
            return n_state, tuple(variable_vector.to_indices())
            
        case _:
            return state, variables
