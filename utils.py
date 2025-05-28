import jax
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as random
from jax import vmap, jacobian

jax.config.update("jax_enable_x64",True)

key = random.key(8765451678)

n = 100
d = 2


def rpchol(A: np.ndarray, k: int):
    F = np.zeros((A.shape[0], k))
    d = np.diag(A)
    S = []

    for i in range(k):

        dist = (d/(np.sum(d))).flatten()

        si = np.random.choice(a = len(dist), p = dist)

        g = A[:, si]

        FF = F[:, :i] @ F[si, :i].T

        g = g - FF

        g_prime = g/(np.sqrt(g[si]) + 1e-14)

        F[:, i] = g_prime

        d = d - F[:, i]**2

        d = d.clip(min = 0)

        S.append(si)

    return F, S

def kernel(gamma):
    def rbf(x, y):
        return jnp.exp(-gamma*jnp.sum((x-y)**2))
    return rbf


"""
    Solves linear least squares $$|fun(x) - y|^2$$ where fun might be nonlinear. Slight regularization. 

"""
def gauss_newton(fun,y, max_iters, x0, tol = 1e-8, reg = 1e-12, verbose = False):
    errs = []
    x = x0
    obj_val = jnp.linalg.norm(fun(x) - y)

    errs.append(obj_val)
    for iter in range(max_iters):
        
        Jac = jacobian(fun)(x)

        x = x + jnp.linalg.solve(Jac.T @ Jac + reg*jnp.eye(len(x)), Jac.T @ (y - fun(x)))

        obj_val = jnp.linalg.norm(fun(x) - y)
        errs.append(obj_val)
        if verbose: print(iter, obj_val)
        if jnp.linalg.norm(fun(x) - y) < tol:
            return x, errs

    return x, errs



if __name__ == "__main__":

    "RUNNING SVD TEST"
    n, d, r = 200, 2, 20

    k = kernel(3)
    vec_k = vmap(vmap(k, in_axes=(None, 0)), in_axes=(0, None))

    X = random.normal(key, shape=(n, d))
    K = vec_k(X, X)
    K = np.array(K)

    F, S = rpchol(K, r)

    svd_errors = []
    rp_errors = []

    for ratio, j in enumerate(np.arange(0.01, 0.95, 0.025)):

        dim = int(j * n)
        u, s, vh = np.linalg.svd(K, hermitian = True)
        s[dim:] = 0
        best_approx = (u * s) @ vh
        F, _ = rpchol(K, dim)

        svd_error = np.linalg.norm(best_approx - K) 
        rp_error = np.linalg.norm(F @ F.T - K)
        svd_errors.append(svd_error)
        rp_errors.append(rp_error)


    plt.plot(svd_errors, label='SVD_Truncation')
    plt.plot(rp_errors, label = "RPCholesky")
    plt.legend()
    plt.grid()
    plt.yscale('log')
    plt.xlabel("Approximation rank")
    plt.ylabel("Frobenius norm error")

    plt.show()

