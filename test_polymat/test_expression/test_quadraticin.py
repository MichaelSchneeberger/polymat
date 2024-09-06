import unittest

from polymat.expressiontree.init import init_from_sparse_repr, init_quadratic_coefficients
from polymat.sparserepr.init import init_from_polynomial_matrix
from polymat.state import init_state



class TestQuadraticIn(unittest.TestCase):

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

        monomial_terms = {
            (0, 0): {
                tuple(): 1.0,            # 1
            },
            (1, 0): {
                ((0, 1),): 1.0,         # x1
            },
            (2, 0): {
                ((0, 1), (1, 1)): 1.0,  # x1 x2
            },
        }

        variable_terms = {
            (0, 0): {((0, 1),): 1},
            (1, 0): {((1, 1),): 1},
        }

        child = init_from_sparse_repr(
            init_from_polynomial_matrix(
                data=child_terms,
                shape=(1, 1),
            )
        )

        monomials = init_from_sparse_repr(
            init_from_polynomial_matrix(
                data=monomial_terms,
                shape=(3, 1),
            )
        )

        variables = init_from_sparse_repr(
            init_from_polynomial_matrix(
                data=variable_terms,
                shape=(2, 1),
            )
        )

        expr = init_quadratic_coefficients(
            child=child,
            monomials=monomials,
            variables=variables,
            stack=tuple()
        )

        state = init_state()
        state, sparse_repr = expr.apply(state)

        data = sparse_repr.at(0, 1)
        self.assertTrue({
            tuple(): 1.0,
            ((2, 1),): 2.0,
        }.items() <= data.items())

        data = sparse_repr.at(1, 1)
        self.assertTrue({
            ((3, 1),): 3.0,
        }.items() <= data.items())

        data = sparse_repr.at(2, 2)
        self.assertTrue({
            ((4, 1),): 4.0,
        }.items() <= data.items())

        data = sparse_repr.at(1, 2)
        self.assertTrue({
            ((5, 1),): 5.0,
        }.items() <= data.items())
    