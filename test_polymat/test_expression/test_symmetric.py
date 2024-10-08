import unittest

from polymat.expressiontree.init import init_from_sparse_repr, init_to_symmetric_matrix
from polymat.sparserepr.init import init_from_polynomial_matrix
from polymat.state import init_state


class TestSymmetric(unittest.TestCase):
    def test_1(self):
        expr_terms = {
            (0, 0): {
                ((0, 1),): 1.0,
            },
            (1, 0): {
                ((1, 1),): 1.0,
            },
            (0, 1): {
                ((1, 1),): 1.0,
                ((2, 1),): 1.0,
            },
            (2, 0): {
                ((1, 1),): 1.0,
            }
        }

        expr = init_from_sparse_repr(
            init_from_polynomial_matrix(
                data=expr_terms,
                shape=(3, 3),
            )
        )

        expr = init_to_symmetric_matrix(expr)

        state = init_state()
        state, sparse_repr = expr.apply(state)

        data = sparse_repr.at(0, 0)
        self.assertTrue(
            {
                ((0, 1),): 1.0,
            }.items()
            <= data.items()
        )

        data = sparse_repr.at(0, 1)
        self.assertTrue(
            {
                ((1, 1),): 1.0,
                ((2, 1),): 0.5,
            }.items()
            <= data.items()
        )

        data = sparse_repr.at(1, 0)
        self.assertTrue(
            {
                ((1, 1),): 1.0,
                ((2, 1),): 0.5,
            }.items()
            <= data.items()
        )

        data = sparse_repr.at(2, 0)
        self.assertTrue(
            {
                ((1, 1),): 0.5,
            }.items()
            <= data.items()
        )

        data = sparse_repr.at(0, 2)
        self.assertTrue(
            {
                ((1, 1),): 0.5,
            }.items()
            <= data.items()
        )
