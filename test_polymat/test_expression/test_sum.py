import unittest

from polymat.expressiontree.init import init_from_sparse_repr, init_sum
from polymat.sparserepr.init import init_from_polynomial_matrix
from polymat.state import init_state


class TestSum(unittest.TestCase):
    def test_1(self):
        expr_terms = {
            (0, 0): {
                tuple(): 2.0,
                ((0, 1),): 3.0,
            },
            (0, 1): {
                tuple(): 1.0,
                ((0, 2),): 1.0,
            },
        }

        expr = init_from_sparse_repr(
            init_from_polynomial_matrix(
                data=expr_terms,
                shape=(2, 2),
            )
        )

        expr = init_sum(expr)

        state = init_state()
        state, sparse_repr = expr.apply(state)

        data = sparse_repr.at(0, 0)
        self.assertTrue(
            {
                tuple(): 3.0,
                ((0, 1),): 3.0,
                ((0, 2),): 1.0,
            }.items()
            <= data.items()
        )
