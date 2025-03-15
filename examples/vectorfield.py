import numpy as np
from matplotlib import pyplot

import polymat
import polymat.typing

x1, x2 = (polymat.define_variable(name) for name in ('x1', 'x2'))
x = polymat.v_stack((x1, x2))

f1 = 0.1*x1**2 + 0.2*x1*x2 + 0.1*x2**2 - 1
f2 = x1*x2 + x1 + x2 - 1

f = f1 * f2

df = f.diff(x)

context = polymat.init_state()

context, df1_array = polymat.to_array(df[0, 0], x).apply(context)
context, df2_array = polymat.to_array(df[0, 1], x).apply(context)

def plot_vector_field():
    pyplot.close()
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.subplots()

    x1 = np.linspace(-2, 2, 20)
    x2 = np.linspace(-2, 2, 20)
    X1, X2 = np.meshgrid(x1, x2)

    U1 = np.vectorize(lambda x1, x2: df1_array(np.array((x1, x2)).reshape(-1, 1)))(X1, X2)
    U2 = np.vectorize(lambda x1, x2: df2_array(np.array((x1, x2)).reshape(-1, 1)))(X1, X2)

    ax.quiver(X1, X2, U1, U2)

    pyplot.show()

plot_vector_field()
