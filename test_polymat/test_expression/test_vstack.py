import unittest

from polymat.expressiontree.init import init_from_sparse_repr, init_v_stack
from polymat.sparserepr.init import init_from_polynomial_matrix
from polymat.state import init_state


class TestVStack(unittest.TestCase):
    def test_1(self):
        left_terms = {
            (0, 0): {
                ((1, 1),): 1.0,
            },
            (1, 0): {
                tuple(): 2.0,
            },
        }

        right_terms = {
            (0, 0): {
                tuple(): 3.0,
            },
            (1, 1): {
                tuple(): 4.0,
            },
        }

        left = init_from_sparse_repr(
            init_from_polynomial_matrix(
                data=left_terms,
                shape=(2, 2),
            )
        )

        right = init_from_sparse_repr(
            init_from_polynomial_matrix(
                data=right_terms,
                shape=(2, 2),
            )
        )

        expr = init_v_stack(children=(left, right), stack=tuple())

        state = init_state()
        state, sparse_repr = expr.apply(state)

        data = sparse_repr.at(0, 0)
        self.assertTrue(
            {
                ((1, 1),): 1.0,
            }.items()
            <= data.items()
        )

        data = sparse_repr.at(1, 0)
        self.assertTrue(
            {
                tuple(): 2.0,
            }.items()
            <= data.items()
        )

        data = sparse_repr.at(2, 0)
        self.assertTrue(
            {
                tuple(): 3.0,
            }.items()
            <= data.items()
        )

        data = sparse_repr.at(3, 1)
        self.assertTrue(
            {
                tuple(): 4.0,
            }.items()
            <= data.items()
        )
