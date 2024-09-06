import unittest

from polymat.expressiontree.init import init_from_sparse_repr, init_linear_coefficients
from polymat.sparserepr.init import init_from_polynomial_matrix
from polymat.state import init_state


class TestQuadraticIn(unittest.TestCase):
    def test_1(self):
        child_terms = {
            (0, 0): {
                ((0, 1),): 2.0,
                ((1, 1),): 3.0,
            },
        }

        monomial_terms = {
            (0, 0): {
                ((0, 1),): 1.0,
            },
            (1, 0): {
                ((2, 1),): 1.0,
            },
            (2, 0): {
                ((1, 1),): 1.0,
            },
            (3, 0): {
                ((3, 1),): 1.0,
            },
        }

        variable_terms = {
            (0, 0): {((0, 1),): 1},
            (1, 0): {((1, 1),): 1},
        }

        child = init_from_sparse_repr(
            init_from_polynomial_matrix(
                data=child_terms,
                shape=(2, 1),
            )
        )

        monomials = init_from_sparse_repr(
            init_from_polynomial_matrix(
                data=monomial_terms,
                shape=(4, 1),
            )
        )

        variables = init_from_sparse_repr(
            init_from_polynomial_matrix(
                data=variable_terms,
                shape=(2, 1),
            )
        )

        expr = init_linear_coefficients(
            child=child, monomials=monomials, variables=variables, stack=tuple()
        )

        state = init_state()
        state, sparse_repr = expr.apply(state)

        data = sparse_repr.at(0, 0)
        self.assertTrue(
            {
                tuple(): 2.0,
            }.items()
            <= data.items()
        )

        data = sparse_repr.at(0, 2)
        self.assertTrue(
            {
                tuple(): 3.0,
            }.items()
            <= data.items()
        )
