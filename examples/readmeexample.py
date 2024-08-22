import polymat

# create the state object
state = polymat.init_state()


# (scalar) polynomial matrix expression
#######################################

names = ('x1', 'x2')
x1, x2 = (polymat.define_variable(n) for n in names)
x = polymat.v_stack((x1, x2))

f1 = x1 + x2
f2 = x1 + x1*x2

f3 = f1 + f2

# prints a nicely printable string representation of the expression
# add(add(x1, x2), add(x1, mul(x1, x2)))
print(f'{f3}')

# prints the string representation of the dataclass
# f3 = ExpressionImpl(
#   child=AdditionExprImpl(
#       left=AdditionExprImpl(
#           left=FromVariableImpl(variable='x1', nvar=1),
#           right=FromVariableImpl(variable='x2', nvar=1)),
#       right=AdditionExprImpl(
#           left=FromVariableImpl(variable='x1', nvar=1),
#           right=ElementwiseMultImpl(
#               left=FromVariableImpl(variable='x1', nvar=1),
#               right=FromVariableImpl(variable='x2', nvar=1)))))
print(f'{f3=}')


# sympy representation
######################

# computes the sympy representation of the expression
state, sympy_repr = polymat.to_sympy(f3,).apply(state)

# prints the sympy representation
# x1*x2 + 2.0*x1 + x2
print(f'{sympy_repr}')


# array representation
######################

# computes the array representation of the expression
state, array_repr = polymat.to_array(f3, x).apply(state)

# prints the array representations
# array_repr.data[1]=array([[2., 1.]])
print(f'{array_repr.data[1]=}')               # numpy array
# array_repr.data[2].toarray()=array([[0. , 0.5, 0.5, 0. ]])
print(f'{array_repr.data[2].toarray()=}')     # sparse scipy array converted to an numpy array
