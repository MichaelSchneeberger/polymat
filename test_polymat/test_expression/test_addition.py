import unittest

from polymat.expressiontree.init import init_from_sparse_repr, init_addition
from polymat.sparserepr.init import init_sparse_repr_from_data
from polymat.state import init_state


class TestAddition(unittest.TestCase):
    def test_1(self):
        left_terms = {
            (0, 0): {
                tuple(): 1.0,
                ((0, 1),): 1.0,
            },
            (1, 0): {
                ((0, 2),): 1.0,
            },
        }

        right_terms = {
            (0, 0): {
                tuple(): 3.0,
                ((1, 1),): 2.0,
            },
            (1, 1): {
                tuple(): 1.0,
            },
        }

        left = init_from_sparse_repr(
            init_sparse_repr_from_data(
                data=left_terms,
                shape=(2, 2),
            )
        )

        right = init_from_sparse_repr(
            init_sparse_repr_from_data(
                data=right_terms,
                shape=(2, 2),
            )
        )

        expr = init_addition(left=left, right=right, stack=tuple())

        state = init_state()
        state, sparse_repr = expr.apply(state)

        data = sparse_repr.at(0, 0)
        self.assertTrue(
            {
                tuple(): 4.0,
                ((0, 1),): 1.0,
                ((1, 1),): 2.0,
            }.items()
            <= data.items()
        )

        data = sparse_repr.at(1, 0)
        self.assertTrue(
            {
                ((0, 2),): 1.0,
            }.items()
            <= data.items()
        )

        data = sparse_repr.at(1, 1)
        self.assertTrue({
            tuple(): 1.0,
        }.items() <= data.items())
