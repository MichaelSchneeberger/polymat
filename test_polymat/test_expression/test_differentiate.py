import unittest

from polymat.expressiontree.init import init_from_sparse_repr, init_differentiate
from polymat.sparserepr.init import init_from_polynomial_matrix
from polymat.state import init_state



class TestDifferentiate(unittest.TestCase):

    def test_1(self):
        child_terms = {
            (0, 0): {
                ((0, 1),): 2.0,
                ((1, 2),): 3.0,
            },
            (1, 0): {
                tuple(): 5.0,
                ((0, 1), (2, 3)): 4.0,
            },
        }

        variable_terms = {
            (0, 0): {((0, 1),): 1},
            (1, 0): {((1, 1),): 1},
            (2, 0): {((2, 1),): 1},
        }

        child = init_from_sparse_repr(
            init_from_polynomial_matrix(
                data=child_terms,
                shape=(2, 1),
            )
        )

        variables = init_from_sparse_repr(
            init_from_polynomial_matrix(
                data=variable_terms,
                shape=(3, 1),
            )
        )

        expr = init_differentiate(
            child=child,
            variables=variables,
            stack=tuple()
        )

        state = init_state()
        state, sparse_repr = expr.apply(state)

        self.assertTupleEqual(sparse_repr.shape, (2, 3))

        data = sparse_repr.at(0, 0)
        self.assertTrue({
            tuple(): 2.0,
        }.items() <= data.items())

        data = sparse_repr.at(0, 1)
        self.assertTrue({
            ((1, 1),): 6.0,
        }.items() <= data.items())

        data = sparse_repr.at(1, 0)
        self.assertTrue({
            ((2, 3),): 4.0,
        }.items() <= data.items())

        data = sparse_repr.at(1, 2)
        self.assertTrue({
            ((0, 1), (2, 2)): 12.0,
        }.items() <= data.items())
    