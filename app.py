import matplotlib.pyplot as plt
import autograd.numpy as np  # Thinly-wrapped version of Numpy
from autograd import grad, elementwise_grad
from autograd import extend

EPS_TOL = 1e-10


def grad_descent(f, df_dx, initial_x):
    alpha = 1.0
    curr_x = initial_x
    #   print("start val: {}".format(f(curr_x)))
    for i in range(10):
        curr_val = f(curr_x)
        for j in range(25):
            grad_x = df_dx(curr_x)
            new_x = curr_x - alpha * grad_x
            new_val = f(new_x)
            if new_val < curr_val:
                curr_x = new_x
                break
            else:
                alpha /= 2.
    curr_grad = df_dx(curr_x)
    #   print("end val: {} | grad_value: {}".format(f(curr_x), curr_grad))
    converged = curr_grad < EPS_TOL
    if not converged:
        print("DID NOT CONVERGE! CHECK")
    return curr_x


if __name__ == '__main__':
    # Data for a three-dimensional line
    def O2(x, y):
        return (3 * x + y) ** 2


    x = np.linspace(-3, 5, 25)
    y = np.linspace(-10, 5, 25)

    X, Y = np.meshgrid(x, y)
    Z1 = O2(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z1, rstride=1, cstride=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


    @extend.primitive
    def argmin_y(x, initial_y):
        f = lambda y: O2(x, y)
        return grad_descent(f, grad(f), initial_y)


    def O1(x, initial_y):
        y = argmin_y(x, initial_y)  # THIS HAS TO CONVERGE!
        return 2 + (y + 5) ** 2


    Z2 = np.empty(X.shape)
    for i in range(len(X)):
        for j in range(len(Y)):
            Z2[i, j] = O1(X[i, j], Y[i, j])
    ax.plot_surface(X, Y, Z2, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')


    def optimize_autograd(initial_x, initial_y):
        f = lambda x: O1(x, initial_y)

        new_x = grad_descent(f, grad(f), initial_x)
        return new_x


    def optimize(initial_x, initial_y):
        f = lambda x: O1(x, initial_y)
        dO2_dy = g = lambda x, y: 2 * (3 * x + y)

        pO1_px = 0
        pO1_py = lambda x, y: 2 * (y + 5)

        pg_py = 2
        pg_px = 6

        def dO1_dx(x):
            y = argmin_y(x, initial_y)
            return pO1_px + pO1_py(x, y) * (1. / pg_py * pg_px)

        new_x = grad_descent(f, dO1_dx, initial_x)
        return new_x


    initial_x = 2.0
    initial_y = 4.0

    print(O1(initial_x, initial_y))
    # print(optimize(4., 6.))
    print(grad(O2)(initial_x, initial_y))

    x_optimal = optimize(initial_x, initial_y)
    y_optimal = argmin_y(x_optimal, initial_y)

    print("x_optimal: {}, y_optmial: {} | val: {}".format(x_optimal, y_optimal, O1(x_optimal, y_optimal)))

    ax.plot3D([x_optimal], [y_optimal], [O1(x_optimal, y_optimal)], 'ro')

    plt.show()

"""
LETS TRY WITH AUTODIFF ?
"""


def argmin_vjp(ans, x):
    """
    This should return the jacobian-vector product
    it should calculate d_ans/dx because the vector contains dloss/dans
    then we get with dloss/dans * dans/dx = dloss/dx which we're actually interested in
    """
    g = elementwise_grad(O2, 1)
    dg_dy = elementwise_grad(g, 1)(x, initial_y)
    dg_dx = elementwise_grad(g, 0)(x, initial_y)
    if np.ndim(dg_dy) == 0:  # we have just simple scalar function so we just have to divide instead of inverse
        return lambda v: v * (1. / dg_dy) * dg_dx

    return lambda v: v * np.matmul(np.linalg.inv(dg_dy), dg_dx)


def optimize_autodiff(initial_x):
    f = lambda x: O1(x)

    new_x = grad_descent(f, grad(f), initial_x)
    return new_x


from autograd.test_util import check_grads


extend.defvjp(argmin_y, argmin_vjp)

x_optimal = optimize_autodiff(initial_x)
y_optimal = argmin_y(x_optimal)

print("x_optimal: {}, y_optmial: {} | val: {}".format(x_optimal, y_optimal, O1(x_optimal)))

