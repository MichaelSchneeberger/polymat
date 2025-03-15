from polymat.expressiontree.nodes import ExpressionNode
from polymat.state.state import State as BaseState


type VariableType[State: BaseState] = ExpressionNode[State] | tuple[int, ...]


def to_indices[State: BaseState](
    state: State, variables: VariableType[State],
) -> tuple[State, tuple[int, ...]]:
    match variables:
        case ExpressionNode():
            n_state, variable_vector = variables.apply(state=state)
            return n_state, tuple(variable_vector.to_indices())
            
        case _:
            return state, variables
