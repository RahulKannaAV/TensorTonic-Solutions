def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    for _ in range(steps):
        fdash_x = 2*a*x0 + b
        x0 = x0 - (lr * fdash_x)

    return x0