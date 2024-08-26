import unittest

import polymat
from polymat.expression.init import init_expression
from polymat.expressiontree.init import init_from_sparse_repr
from polymat.sparserepr.init import init_from_polynomial_matrix
from polymat.state import init_state



class TestToArray(unittest.TestCase):

    def test_1(self):
        child_terms = {
            (0, 0): {
                tuple(): 1.0,
                ((1, 1),): 2.0,
            },
            (1, 0): {
                ((0, 1),): 4.0,
                ((0, 1), (1, 1)): 3.0,
                ((1, 2),): 5.0,
            },
            (2, 0): {
                ((0, 1), (1, 2)): 3.0,
            },
        }

        variable_terms = {
            (0, 0): {((0, 1),): 1},
            (1, 0): {((1, 1),): 1},
        }

        expr = init_expression(
                init_from_sparse_repr(
                init_from_polynomial_matrix(
                    data=child_terms, 
                    shape=(3, 1)
                )
            )
        )

        variables = init_expression(
            init_from_sparse_repr(
                init_from_polynomial_matrix(
                    data=variable_terms,
                    shape=(2, 1),
                )
            )
        ).to_variable_vector()

        state = init_state()

        state, result = polymat.to_array(
            expr, 
            variables,
        ).apply(state)

        A0 = result.data[0]
        A1 = result.data[1]
        A2 = result.data[2]
        A3 = result.data[3]

        self.assertEqual(A0[0, 0], 1.0)

        self.assertEqual(A1[0, 1], 2.0)
        self.assertEqual(A1[1, 0], 4.0)

        self.assertEqual(A2[1, 1], 1.5)
        self.assertEqual(A2[1, 2], 1.5)
        self.assertEqual(A2[1, 3], 5.0)

        self.assertEqual(A3[2, 3], 1.0)
        self.assertEqual(A3[2, 5], 1.0)
        self.assertEqual(A3[2, 6], 1.0)
