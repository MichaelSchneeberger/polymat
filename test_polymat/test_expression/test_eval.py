import unittest

from polymat.expressiontree.init import init_from_sparse_repr, init_eval
from polymat.sparserepr.init import init_sparse_repr_from_data
from polymat.state import init_state
from polymat.symbol import Symbol


class TestEval(unittest.TestCase):
    def test_1(self):
        expr_terms = {
            (0, 0): {
                ((0, 1), (2, 1)): 2.0,
                ((0, 1), (1, 1), (3, 1)): 3.0,
            },
            (1, 0): {
                tuple(): 1.0,
                ((1, 2),): 1.0,
                ((2, 1),): 1.0,
            },
        }

        expr = init_from_sparse_repr(
            init_sparse_repr_from_data(
                data=expr_terms,
                shape=(2, 1),
            )
        )

        state = init_state()

        x1 = Symbol("x1")
        x2 = Symbol("x2")

        state, _ = state.register(x1, 1, stack=tuple())
        state, _ = state.register(x2, 1, stack=tuple())

        expr = init_eval(expr, substitutions={x1: (2.0,), x2: (3.0,)}, stack=tuple())

        state, sparse_repr = expr.apply(state)

        data = sparse_repr.at(0, 0)
        self.assertTrue(
            {
                ((2, 1),): 4.0,
                ((3, 1),): 18.0,
            }.items()
            <= data.items()
        )

        data = sparse_repr.at(1, 0)
        self.assertTrue(
            {
                tuple(): 10.0,
                ((2, 1),): 1,
            }.items()
            <= data.items()
        )
