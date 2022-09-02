import numpy as np
import torch
import torch.cuda
from scipy.stats import cauchy
from sklearn.kernel_approximation import PolynomialCountSketch

from cython_backend import sig_kernel_batch_varpar, sig_kernel_Gram_varpar, sig_kernel_Gram_varpar_const
from numba import cuda

from .cuda_backend import compute_sig_kernel_batch_varpar_from_increments_cuda


# ===========================================================================================================
# Static kernels
# ===========================================================================================================
class LinearKernel():
    """Linear kernel k: R^d x R^d -> R"""

    def batch_kernel(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^i_t) of shape (batch, length_X, length_Y)
        """
        return torch.bmm(X, Y.permute(0, 2, 1))

    def Gram_matrix(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        return torch.einsum('ipk,jqk->ijpq', X, Y)


class RFFKernel_1:
    def __init__(self, rff_features, sigma, metric, data_shape, use_offset, linearize_data=False):
        self.use_offset = use_offset
        self.metric = metric
        self.sigma = sigma
        self.rff_features = int(rff_features)
        self.linearize_data = linearize_data
        if not self.linearize_data:
            self.dimensions = data_shape[-1]
        else:
            self.dimensions = np.prod(data_shape[1:])
        if self.sigma == 'auto':
            self.sigma = 1.0 / data_shape[1]

        self.b, self.w = self.fit()

    def fit(self):
        if self.metric == "rbf":
            w = np.sqrt(2 * self.sigma) * np.random.normal(size=(self.rff_features, self.dimensions))
        elif self.metric == "laplace":
            w = cauchy.rvs(scale=self.sigma, size=(self.rff_features, self.dimensions))
        else:
            raise ValueError("Only laplace and rbf kernels are supported for RFF")
        b = 2 * np.pi * np.random.rand(self.rff_features)
        return torch.from_numpy(b), torch.from_numpy(w)

    def transform(self, data):
        result = matrix_mult(data, self.w, transpose_y=True)
        if self.use_offset:
            result += self.b[np.newaxis, :]
            result = np.cos(result)
        else:
            result = torch.tensor(np.hstack((np.cos(result), np.sin(result))))
        return np.sqrt(2 / self.rff_features) * result

    def Gram_matrix(self, X, Y):
        if not self.linearize_data:
            x_z = self.transform(X)
            y_z = self.transform(Y)
            return torch.einsum('ipk,jqk->ijpq', x_z, y_z)
        else:
            x_z = self.transform(torch.flatten(X, start_dim=1))
            y_z = self.transform(torch.flatten(Y, start_dim=1))
            return torch.einsum('ip,jq->ijpq', x_z, y_z)


class RFFKernel:

    def __init__(self, length, gamma=1, dims=10, metric="rbf"):
        self.length = length
        self.metric = metric
        self.dims = dims
        self.gamma = gamma

        self.u, self.w = self._fit()

    def transform(self, X):
        """ Transforms the data X (n_samples, n_features) to the new map space Z(X) (n_samples, n_components)"""
        # Compute feature map Z(x):
        Z = np.sqrt(2 / self.dims) * np.cos((X.mm(self.w.T) + self.u[np.newaxis, :]))
        return Z

    # def Gram_matrix(self, X, Y):
    #     """Input:
    #               - X: torch tensor of shape (batch_X, length_X, dim),
    #               - Y: torch tensor of shape (batch_Y, length_Y, dim)
    #        Output:
    #               - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
    #
    #     X_z = torch.stack([self.transform(x) for x in X])
    #     Y_z = torch.stack([self.transform(y) for y in Y])
    #
    #     """
    #     Xs = self.transform(torch.tensor(vectorise_data(X.detach().numpy())))
    #     Ys = self.transform(torch.tensor(vectorise_data(Y.detach().numpy())))
    #     return torch.einsum('ip,jq->ijpq', Xs, Ys)
    #
    def Gram_matrix(self, X, Y):
        """Input:
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output:
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        self.length = X.shape[0]
        self.u, self.w = self._fit()

        Xs = np.sqrt(2 / self.dims) * torch.einsum('ipk,ji->jpk', X, self.w)  + self.u[ :, np.newaxis,np.newaxis]
        Ys = np.sqrt(2 / self.dims) * torch.einsum('ipk,ji->jpk', Y, self.w)  + self.u[ :, np.newaxis,np.newaxis]
        return torch.einsum('ipk,jqk->ijpq', Xs, Ys)

    def _fit(self):
        if self.metric == "rbf":
            w = np.sqrt(2 * self.gamma) * np.random.normal(size=(self.dims, self.length))
        elif self.metric == "laplace":
            w = cauchy.rvs(scale=self.gamma, size=(self.dims, self.length))

        # Generate D iid samples from Uniform(0,2*pi)
        u = 2 * np.pi * np.random.rand(self.dims)
        return torch.from_numpy(u), torch.from_numpy(w)

def vectorise_data(data):
    return np.array([data[n, :, :].reshape((data.shape[1] * data.shape[2], )) for n in range(data.shape[0])])


def matrix_mult(x, y=None, transpose_x=False, transpose_y=False):
    subscript_x = '...ji' if transpose_x else '...ij'
    subscript_y = '...kj' if transpose_y else '...jk'
    return torch.einsum(f'{subscript_x},{subscript_y}->...ik', x, y if y is not None else x)


class RBFKernel():
    """RBF kernel k: R^d x R^d -> R"""

    def __init__(self, sigma):
        self.sigma = sigma

    def batch_kernel(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^i_t) of shape (batch, length_X, length_Y)
        """
        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        Xs = torch.sum(X ** 2, dim=2)
        Ys = torch.sum(Y ** 2, dim=2)
        dist = -2. * torch.bmm(X, Y.permute(0, 2, 1))
        dist += torch.reshape(Xs, (A, M, 1)) + torch.reshape(Ys, (A, 1, N))
        return torch.exp(-dist / self.sigma)

    def Gram_matrix(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        Xs = torch.sum(X ** 2, dim=2)
        Ys = torch.sum(Y ** 2, dim=2)
        dist = -2. * torch.einsum('ipk,jqk->ijpq', X, Y)
        dist += torch.reshape(Xs, (A, 1, M, 1)) + torch.reshape(Ys, (1, B, 1, N))
        return torch.exp(-dist / self.sigma)


class LaplaceKernel:
    """Laplace kernel k: R^d x R^d -> R"""

    def __init__(self, sigma):
        self.sigma = sigma

    def Gram_matrix(self, X, Y):
        """Input:
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output:
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        r = torch.sum(np.abs(X[:, None, :, None] - Y[None, :, None, :]), axis=4)
        return torch.exp(-r / self.sigma)

# ===========================================================================================================


# ===========================================================================================================
# Signature Kernel
# ===========================================================================================================
class SigKernel():
    """Wrapper of the signature kernel k_sig(x,y) = <S(f(x)),S(f(y))> where k(x,y) = <f(x),f(y)> is a given static kernel"""

    def __init__(self, finite_diff_impl, static_kernel, dyadic_order, _naive_solver=False):
        self.finite_diff_impl = finite_diff_impl
        self.static_kernel = static_kernel
        self.dyadic_order = dyadic_order
        self._naive_solver = _naive_solver

    def compute_kernel(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - vector k(X^i_T,Y^i_T) of shape (batch,)
        """
        return _SigKernel.apply(X, Y, self.static_kernel, self.dyadic_order, self._naive_solver)

    def compute_Gram(self, X, Y, sym=False):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_T,Y^j_T) of shape (batch_X, batch_Y)
        """
        return _SigKernelGram.apply(X, Y, self.finite_diff_impl, self.static_kernel, self.dyadic_order, sym,
                                    self._naive_solver)

    def compute_distance(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - vector ||S(X^i)_T - S(Y^i)_T||^2 of shape (batch,)
        """

        assert not Y.requires_grad, "the second input should not require grad"

        k_XX = self.compute_kernel(X, X)
        k_YY = self.compute_kernel(Y, Y)
        k_XY = self.compute_kernel(X, Y)

        return torch.mean(k_XX) + torch.mean(k_YY) - 2. * torch.mean(k_XY)

    def compute_mmd(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - scalar: MMD signature distance between samples X and samples Y
        """

        assert not Y.requires_grad, "the second input should not require grad"

        K_XX = self.compute_Gram(X, X, sym=True)
        K_YY = self.compute_Gram(Y, Y, sym=True)
        K_XY = self.compute_Gram(X, Y, sym=False)

        return torch.mean(K_XX) + torch.mean(K_YY) - 2. * torch.mean(K_XY)


class _SigKernel(torch.autograd.Function):
    """Signature kernel k_sig(x,y) = <S(f(x)),S(f(y))> where k(x,y) = <f(x),f(y)> is a given static kernel"""

    @staticmethod
    def forward(ctx, X, Y, static_kernel, dyadic_order, _naive_solver=False):

        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2 ** dyadic_order) * (M - 1)
        NN = (2 ** dyadic_order) * (N - 1)

        # computing dsdt k(X^i_s,Y^i_t)
        G_static = static_kernel.batch_kernel(X, Y)
        G_static_ = G_static[:, 1:, 1:] + G_static[:, :-1, :-1] - G_static[:, 1:, :-1] - G_static[:, :-1, 1:]
        G_static_ = tile(
            tile(G_static_, 1, 2 ** dyadic_order) / float(2 ** dyadic_order), 2,
                         2 ** dyadic_order) / float(2 ** dyadic_order)

        # if on GPU
        if X.device.type == 'cuda':

            assert max(MM + 1,
                       NN + 1) < 1024, 'n must be lowered or data must be moved to CPU as the current choice of n makes exceed the thread limit'

            # cuda parameters
            threads_per_block = max(MM + 1, NN + 1)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Prepare the tensor of output solutions to the PDE (forward)
            K = torch.zeros((A, MM + 2, NN + 2), device=G_static.device, dtype=G_static.dtype)
            K[:, 0, :] = 1.
            K[:, :, 0] = 1.

            # Compute the forward signature kernel
            compute_sig_kernel_batch_varpar_from_increments_cuda[A, threads_per_block](
                cuda.as_cuda_array(G_static_.detach()),
                MM + 1, NN + 1, n_anti_diagonals,
                cuda.as_cuda_array(K), _naive_solver)
            K = K[:, :-1, :-1]

        # if on CPU
        else:
            K = torch.tensor(sig_kernel_batch_varpar(G_static_.detach().numpy(), _naive_solver), dtype=G_static.dtype,
                             device=G_static.device)

        ctx.save_for_backward(X, Y, G_static, K)
        ctx.static_kernel = static_kernel
        ctx.dyadic_order = dyadic_order
        ctx._naive_solver = _naive_solver

        return K[:, -1, -1]

    @staticmethod
    def backward(ctx, grad_output):

        X, Y, G_static, K = ctx.saved_tensors
        static_kernel = ctx.static_kernel
        dyadic_order = ctx.dyadic_order
        _naive_solver = ctx._naive_solver

        G_static_ = G_static[:, 1:, 1:] + G_static[:, :-1, :-1] - G_static[:, 1:, :-1] - G_static[:, :-1, 1:]
        G_static_ = tile(tile(G_static_, 1, 2 ** dyadic_order) / float(2 ** dyadic_order), 2,
                         2 ** dyadic_order) / float(2 ** dyadic_order)

        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2 ** dyadic_order) * (M - 1)
        NN = (2 ** dyadic_order) * (N - 1)

        # Reverse paths
        X_rev = torch.flip(X, dims=[1])
        Y_rev = torch.flip(Y, dims=[1])

        # computing dsdt k(X_rev^i_s,Y_rev^i_t) for variation of parameters
        G_static_rev = flip(flip(G_static_, dim=1), dim=2)

        # if on GPU
        if X.device.type == 'cuda':

            # Prepare the tensor of output solutions to the PDE (backward)
            K_rev = torch.zeros((A, MM + 2, NN + 2), device=G_static_rev.device, dtype=G_static_rev.dtype)
            K_rev[:, 0, :] = 1.
            K_rev[:, :, 0] = 1.

            # cuda parameters
            threads_per_block = max(MM, NN)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Compute signature kernel for reversed paths
            compute_sig_kernel_batch_varpar_from_increments_cuda[A, threads_per_block](
                cuda.as_cuda_array(G_static_rev.detach()),
                MM + 1, NN + 1, n_anti_diagonals,
                cuda.as_cuda_array(K_rev), _naive_solver)

            K_rev = K_rev[:, :-1, :-1]

            # if on CPU
        else:
            K_rev = torch.tensor(sig_kernel_batch_varpar(G_static_rev.detach().numpy(), _naive_solver),
                                 dtype=G_static.dtype, device=G_static.device)

        K_rev = flip(flip(K_rev, dim=1), dim=2)
        KK = K[:, :-1, :-1] * K_rev[:, 1:, 1:]

        # finite difference step 
        h = 1e-9

        Xh = X[:, :, :, None] + h * torch.eye(D, dtype=X.dtype, device=X.device)[None, None, :]
        Xh = Xh.permute(0, 1, 3, 2)
        Xh = Xh.reshape(A, M * D, D)

        G_h = static_kernel.batch_kernel(Xh, Y)
        G_h = G_h.reshape(A, M, D, N)
        G_h = G_h.permute(0, 1, 3, 2)

        Diff_1 = G_h[:, 1:, 1:, :] - G_h[:, 1:, :-1, :] - (G_static[:, 1:, 1:])[:, :, :, None] + (G_static[:, 1:, :-1])[
                                                                                                 :, :, :, None]
        Diff_1 = tile(tile(Diff_1, 1, 2 ** dyadic_order) / float(2 ** dyadic_order), 2, 2 ** dyadic_order) / float(
            2 ** dyadic_order)
        Diff_2 = G_h[:, 1:, 1:, :] - G_h[:, 1:, :-1, :] - (G_static[:, 1:, 1:])[:, :, :, None] + (G_static[:, 1:, :-1])[
                                                                                                 :, :, :, None]
        Diff_2 += - G_h[:, :-1, 1:, :] + G_h[:, :-1, :-1, :] + (G_static[:, :-1, 1:])[:, :, :, None] - (G_static[:, :-1,
                                                                                                        :-1])[:, :, :,
                                                                                                       None]
        Diff_2 = tile(tile(Diff_2, 1, 2 ** dyadic_order) / float(2 ** dyadic_order), 2, 2 ** dyadic_order) / float(
            2 ** dyadic_order)

        grad_1 = (KK[:, :, :, None] * Diff_1) / h
        grad_2 = (KK[:, :, :, None] * Diff_2) / h

        grad_1 = torch.sum(grad_1, axis=2)
        grad_1 = torch.sum(grad_1.reshape(A, M - 1, 2 ** dyadic_order, D), axis=2)
        grad_2 = torch.sum(grad_2, axis=2)
        grad_2 = torch.sum(grad_2.reshape(A, M - 1, 2 ** dyadic_order, D), axis=2)

        grad_prev = grad_1[:, :-1, :] + grad_2[:, 1:, :]  # /¯¯
        grad_next = torch.cat([torch.zeros((A, 1, D), dtype=X.dtype, device=X.device), grad_1[:, 1:, :]], dim=1)  # /
        grad_incr = grad_prev - grad_1[:, 1:, :]
        grad_points = torch.cat(
            [(grad_2[:, 0, :] - grad_1[:, 0, :])[:, None, :], grad_incr, grad_1[:, -1, :][:, None, :]], dim=1)

        if Y.requires_grad:
            grad_points *= 2

        return grad_output[:, None, None] * grad_points, None, None, None, None


class _SigKernelGram(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y, fin_diff_impl, static_kernel, dyadic_order, sym=False, _naive_solver=False):

        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2 ** dyadic_order) * (M - 1)
        NN = (2 ** dyadic_order) * (N - 1)

        # computing dsdt k(X^i_s,Y^j_t)
        G_static = static_kernel.Gram_matrix(X, Y)
        G_static_ = G_static[:, :, 1:, 1:] + G_static[:, :, :-1, :-1] - G_static[:, :, 1:, :-1] - G_static[:, :, :-1,
                                                                                                  1:]
        G_static_ = tile(tile(G_static_, 2, 2 ** dyadic_order) / float(2 ** dyadic_order), 3,
                         2 ** dyadic_order) / float(2 ** dyadic_order)

        G = torch.tensor(
            fin_diff_impl(G_static_.detach().numpy(), sym, _naive_solver),
            dtype=G_static.dtype, device=G_static.device
        )

        ctx.save_for_backward(X, Y, G, G_static)
        ctx.sym = sym
        ctx.static_kernel = static_kernel
        ctx.fin_diff_impl = fin_diff_impl
        ctx.dyadic_order = dyadic_order
        ctx._naive_solver = _naive_solver

        return G[:, :, -1, -1]

    @staticmethod
    def backward(ctx, grad_output):

        X, Y, G, G_static = ctx.saved_tensors
        sym = ctx.sym
        static_kernel = ctx.static_kernel
        fin_diff_impl = ctx.fin_diff_impl
        dyadic_order = ctx.dyadic_order
        _naive_solver = ctx._naive_solver

        G_static_ = G_static[:, :, 1:, 1:] + G_static[:, :, :-1, :-1] - G_static[:, :, 1:, :-1] - G_static[:, :, :-1,
                                                                                                  1:]
        G_static_ = tile(tile(G_static_, 2, 2 ** dyadic_order) / float(2 ** dyadic_order), 3,
                         2 ** dyadic_order) / float(2 ** dyadic_order)

        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2 ** dyadic_order) * (M - 1)
        NN = (2 ** dyadic_order) * (N - 1)

        # Reverse paths
        X_rev = torch.flip(X, dims=[1])
        Y_rev = torch.flip(Y, dims=[1])

        # computing dsdt k(X_rev^i_s,Y_rev^j_t) for variation of parameters
        G_static_rev = flip(flip(G_static_, dim=2), dim=3)

        G_rev = torch.tensor(
            fin_diff_impl(G_static_rev.detach().numpy(), sym, _naive_solver),
            dtype=G_static.dtype, device=G_static.device
        )

        G_rev = flip(flip(G_rev, dim=2), dim=3)
        GG = G[:, :, :-1, :-1] * G_rev[:, :, 1:, 1:]

        # finite difference step 
        h = 1e-9

        Xh = X[:, :, :, None] + h * torch.eye(D, dtype=X.dtype, device=X.device)[None, None, :]
        Xh = Xh.permute(0, 1, 3, 2)
        Xh = Xh.reshape(A, M * D, D)

        G_h = static_kernel.Gram_matrix(Xh, Y)
        G_h = G_h.reshape(A, B, M, D, N)
        G_h = G_h.permute(0, 1, 2, 4, 3)

        Diff_1 = G_h[:, :, 1:, 1:, :] - \
                 G_h[:, :, 1:, :-1, :] - \
                 (G_static[:, :, 1:, 1:])[:, :, :, :, None] + \
                 (G_static[:, :, 1:, :-1])[:, :, :, :, None]
        Diff_1 = tile(tile(Diff_1, 2, 2 ** dyadic_order) / float(2 ** dyadic_order), 3, 2 ** dyadic_order) / float(
            2 ** dyadic_order)
        Diff_2 = G_h[:, :, 1:, 1:, :] - \
                 G_h[:, :, 1:, :-1, :] - \
                 (G_static[:, :, 1:, 1:])[:, :, :, :, None] + \
                 (G_static[:, :, 1:, :-1])[:, :, :, :, None]
        Diff_2 += -G_h[:, :, :-1, 1:, :] + \
                  G_h[:, :, :-1, :-1, :] + \
                  (G_static[:, :, :-1, 1:])[:, :, :, :, None] - \
                  (G_static[:, :, :-1, :-1])[:, :, :, :, None]
        Diff_2 = tile(tile(Diff_2, 2, 2 ** dyadic_order) / float(2 ** dyadic_order), 3, 2 ** dyadic_order) / float(
            2 ** dyadic_order)

        grad_1 = (GG[:, :, :, :, None] * Diff_1) / h
        grad_2 = (GG[:, :, :, :, None] * Diff_2) / h

        grad_1 = torch.sum(grad_1, axis=3)
        grad_1 = torch.sum(grad_1.reshape(A, B, M - 1, 2 ** dyadic_order, D), axis=3)
        grad_2 = torch.sum(grad_2, axis=3)
        grad_2 = torch.sum(grad_2.reshape(A, B, M - 1, 2 ** dyadic_order, D), axis=3)

        grad_prev = grad_1[:, :, :-1, :] + grad_2[:, :, 1:, :]  # /¯¯
        grad_next = torch.cat([torch.zeros((A, B, 1, D), dtype=X.dtype, device=X.device), grad_1[:, :, 1:, :]],
                              dim=2)  # /
        grad_incr = grad_prev - grad_1[:, :, 1:, :]
        grad_points = torch.cat(
            [(grad_2[:, :, 0, :] - grad_1[:, :, 0, :])[:, :, None, :], grad_incr, grad_1[:, :, -1, :][:, :, None, :]],
            dim=2)

        if sym:
            grad = (grad_output[:, :, None, None] * grad_points + grad_output.t()[:, :, None, None] * grad_points).sum(
                dim=1)
            return grad, None, None, None, None, None
        else:
            grad = (grad_output[:, :, None, None] * grad_points).sum(dim=1)
            return grad, None, None, None, None, None


# ===========================================================================================================

# ===========================================================================================================
# Finite Diff Impls
# ===========================================================================================================
# standard quadrature with w(x) = 1 and limits from -1 t0 1
quad_w_x_16_standard = [
    (0.1894506104550685, -0.0950125098376374),
    (0.1894506104550685, 0.0950125098376374),
    (0.1826034150449236, -0.2816035507792589),
    (0.1826034150449236, 0.2816035507792589),
    (0.1691565193950025, -0.4580167776572274),
    (0.1691565193950025, 0.4580167776572274),
    (0.1495959888165767, -0.6178762444026438),
    (0.1495959888165767, 0.6178762444026438),
    (0.1246289712555339, -0.7554044083550030),
    (0.1246289712555339, 0.7554044083550030),
    (0.0951585116824928, -0.8656312023878318),
    (0.0951585116824928, 0.8656312023878318),
    (0.0622535239386479, -0.9445750230732326),
    (0.0622535239386479, 0.9445750230732326),
    (0.0271524594117541, -0.9894009349916499),
    (0.0271524594117541, 0.9894009349916499),
]

# https://reader.elsevier.com/reader/sd/pii/0021999181900991?token=0A7DC74E3914BF9665B9FFED1E759F355C0292BE74919C13D6ECDAF2FC3C72008DFA8BA7189615B6AD6A0CFDAD53B2A5&originRegion=eu-west-1&originCreation=20220626174347
# w(x) = xe^{-^2}, from 0 to inf
quad_w_x_16 = [
    # weight - X
    (0.3795307814831678e-2, 0.4775799543737674e-1),
    (0.2136808301992049e-1, 0.1575643611266753),
    (0.5595857089379011e-1, 0.3236556568455920),
    (0.9587168277747507e-1, 0.5391473546675038),
    (0.1169082070371872, 0.7970053979972014),
    (0.1029363012162623, 0.1090958307363892e1),
    (0.6468246716393942e-1, 0.1415975970714936e1),
    (0.2831911613754905e-1, 0.1768437030466615e1),
    (0.8362647991652432e-2, 0.2146149962010079e1),
    (0.1597736202726321e-2, 0.2548365652625752e1),
    (0.1870134647150351e-3, 0.2975896592510777e1),
    (0.1243935496206526e-4, 0.3431483868308089e1),
    (0.4208466925294357e-6, 0.3920694119664905e1),
    (0.6051847030054333e-8, 0.4454120573510955e1),
    (0.2643406562982473e-10, 0.5053674269642785e1),
    (0.1524594098604790e-13, 0.5778478847939104e1),
]

# https://math.stackexchange.com/questions/1544918/gaussian-quadrature-with-a-to-0-1-reference-domain-instead-of-a-1-1-ref
# fixme: original weights scaled for the interval
quad_w_x_16_0_1 = [(w / 2., (x + 1) / 2.) for w, x in quad_w_x_16_standard]


# ===========================================================================================================
# Various utility functions
# ===========================================================================================================
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:,
        getattr(torch.arange(x.size(1) - 1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


# ===========================================================================================================
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
        a.device)
    return torch.index_select(a, dim, order_index)
# ===========================================================================================================
