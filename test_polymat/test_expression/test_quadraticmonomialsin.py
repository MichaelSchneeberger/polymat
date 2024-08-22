import unittest

from polymat.expressiontree.init import init_from_sparse_repr, init_quadratic_monomials
from polymat.sparserepr.init import init_sparse_repr_from_data
from polymat.state import init_state



class TestQuadraticMonomialsIn(unittest.TestCase):

    def test_1(self):
        child_terms = {
            (0, 0): {
                ((0, 1),): 1.0,  # x1
                ((0, 1), (2, 1)): 2.0,  # x1 
                ((0, 2), (3, 1)): 3.0,  # x1 x1
                ((0, 2), (1, 2), (4, 1)): 4.0,  # x1 x1 x2 x2
                ((0, 2), (1, 1), (5, 1)): 5.0,  # x1 x1 x2
            }
        }
        variable_terms = {
            (0, 0): {((0, 1),): 1},
            (1, 0): {((1, 1),): 1},
        }

        child = init_from_sparse_repr(
            init_sparse_repr_from_data(
                data=child_terms,
                shape=(1, 1),
            )
        )

        variables = init_from_sparse_repr(
            init_sparse_repr_from_data(
                data=variable_terms,
                shape=(2, 1),
            )
        )

        expr = init_quadratic_monomials(
            child=child,
            variables=variables,
        )

        state = init_state()
        state, sparse_repr = expr.apply(state)

        data = sparse_repr.at(0, 0)
        self.assertTrue({
            tuple(): 1.0,
        }.items() <= data.items())

        data = sparse_repr.at(1, 0)
        self.assertTrue({
            ((0, 1),): 1.0,
        }.items() <= data.items())

        data = sparse_repr.at(2, 0)
        self.assertTrue({
            ((0, 1), (1, 1)): 1.0,
        }.items() <= data.items())
