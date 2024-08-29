import polymat

# The state object is passed through all operations involved in a polynomial
# expression
state = polymat.init_state()


# polynomial expression
#######################

# Define the polynomial variables and stack them into one vector
names = ('x1', 'x2')
x1, x2 = (polymat.define_variable(n) for n in names)
x = polymat.v_stack((x1, x2))

# Create a polynomial expression using arithmetic operations
f = (x1 + x2) + (x1 + x1*x2)

# Prints a pretty string representation of the polynomial expression:
# add(add(x1, x2), add(x1, mul(x1, x2)))
print(f'{f}')

# Prints the Python implementation of the expression using dataclasses:
# f3 = ExpressionImpl(
#   child=AdditionExprImpl(
#       left=AdditionExprImpl(
#           left=FromVariableImpl(variable='x1', size=1),
#           right=FromVariableImpl(variable='x2', size=1)),
#       right=AdditionExprImpl(
#           left=FromVariableImpl(variable='x1', size=1),
#           right=ElementwiseMultImpl(
#               left=FromVariableImpl(variable='x1', size=1),
#               right=FromVariableImpl(variable='x2', size=1)))))
print(f'{f=}')


# sympy representation
######################

# Compute the sympy representation of the expression
state, sympy_repr = polymat.to_sympy(f,).apply(state)

# Prints the sympy representation:
# x1*x2 + 2.0*x1 + x2
print(f'{sympy_repr}')


# array representation
######################

# Compute the array representation of the expression
state, array_repr = polymat.to_array(f, x).apply(state)

# Prints the array representations:
# array_repr.data[1]=array([[2., 1.]])
print(f'{array_repr.data[1]=}')               # numpy array
# array_repr.data[2].toarray()=array([[0. , 0.5, 0.5, 0. ]])
print(f'{array_repr.data[2].toarray()=}')     # sparse scipy array converted to an numpy array