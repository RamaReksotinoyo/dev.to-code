from typing import Callable, Sequence, Union
import itertools
import numpy as np


DifferentiableFunc = Callable[[np.ndarray], np.ndarray]


def finite_diff(
        f: DifferentiableFunc,
        args: Sequence[np.ndarray],
        eps: float = 1e-7,
    ) -> Sequence[np.ndarray]:
    """
    Compute the gradient of f with respect to each argument in args.
    """
    grads = []
    for _, arg in enumerate(args):
        grad = np.empty_like(arg) # allocate space for the gradient
        for ix in itertools.product(*[range(x) for x in arg.shape]): # iterate over all indices
            v = arg[ix].copy() # save the original value
            arg[ix] = v + eps
            v1 = float(f(*args))  # must be scalar
            arg[ix] = v - eps
            v2 = float(f(*args)) # must be scalar
            arg[ix] = v
            grad[ix] = (v1-v2) / (2*eps) # the slope
        grads.append(grad)
    return grads


if '__main__' == __name__:
    x = np.arange(8, dtype=float).reshape(2, 4)
    assert np.allclose(finite_diff(lambda x: (x**2).sum(), [x]), [2 * x])
    print(finite_diff(lambda x: (x**3).sum(), [x]))


# [kkurniawan](https://kkurniawan.com/blog/autodiff/)
# [rhome](https://medium.com/@rhome/automatic-differentiation-26d5a993692b)
# [ari seff](https://www.youtube.com/watch?v=wG_nF1awSSY)
