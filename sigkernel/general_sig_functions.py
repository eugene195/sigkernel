import numpy as np

from sigkernel import sig_kernel_Gram_varpar, sig_kernel_Gram_varpar_const, quad_w_x_16, quad_w_x_16_0_1


def benchmark_finite_diff_impl(_lambda=None):
    return lambda G, sym, _naive_solver: sig_kernel_Gram_varpar(G, sym, _naive_solver)


def const_weight_kernel(_lambda):
    return lambda G, sym, _naive_solver: sig_kernel_Gram_varpar_const(G, _lambda, sym, _naive_solver)


def const_exp_kernel(_lambda):
    return const_weight_kernel(np.exp(_lambda))

#
# def rayleigh_rv_quad(_lambda=None):
#     def quad_impl(G, sym, _naive_solver):
#         return 2 * sum([
#             weight * sig_kernel_Gram_varpar_const(G, x, sym, _naive_solver)
#             for weight, x in quad_w_x_16  # quad_w_x_3
#         ])
#
#     return lambda G, sym, _naive_solver: quad_impl(G, sym, _naive_solver)

import concurrent.futures


def worker(param):
    weight_, x_, G_, sym_, _naive_solver_ = param
    print("weight: {}, x: {}".format(weight_, x_))
    return weight_ * sig_kernel_Gram_varpar_const(G_, x_, sym_, _naive_solver_)


def rayleigh_rv_quad(_lambda=None):
    def quad_impl(G, sym, _naive_solver):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(worker, (weight, x, G, sym, _naive_solver))
                for weight, x in quad_w_x_16
            ]
            return 2 * sum((future.result() for future in concurrent.futures.as_completed(futures)))
    return lambda G, sym, _naive_solver: quad_impl(G, sym, _naive_solver)


def uniform_rv_quad(_lambda=None):
    def quad_impl(G, sym, _naive_solver):
        return sum([
            weight * sig_kernel_Gram_varpar_const(G, x, sym, _naive_solver)
            for weight, x in quad_w_x_16_0_1
        ])

    return lambda G, sym, _naive_solver: quad_impl(G, sym, _naive_solver)