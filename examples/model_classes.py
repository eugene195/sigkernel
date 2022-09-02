import numpy as np

import sigkernel
from examples.global_config import add_time_axis, scale_transform, add_lead_lag, PDE_LAMBDAS, rbf_sigma, rff_metric, \
    rff_use_offset
from sigkernel.general_sig_functions import benchmark_finite_diff_impl, rayleigh_rv_quad, \
    uniform_rv_quad, const_exp_kernel


class BaseModel:
    def __init__(self, model_impl, model_name):
        self.model_name = model_name
        self.model_impl = model_impl
        self.model_params = {
            "add_time_axis": add_time_axis,
            "scale_transform": scale_transform,
            "add_lead_lag": add_lead_lag,
        }

    def override_params(self, param_name, param_values):
        self.model_params[param_name] = param_values


class BenchmarkModel(BaseModel):
    def __init__(self):
        super().__init__(benchmark_finite_diff_impl, "benchmark")

    def get_model_impl(self, params_dict=None):
        return self.model_impl(_lambda=1.0)

    def get_params(self):
        return self.model_params


class ConstScalingModel(BaseModel):
    def __init__(self):
        super().__init__(const_exp_kernel, "const")
        self.model_params["pde_lambda"] = PDE_LAMBDAS

    def get_model_impl(self, params_dict):
        return self.model_impl(_lambda=params_dict["pde_lambda"])

    def get_params(self):
        return self.model_params


class RayleighRVModel(BaseModel):
    def __init__(self):
        super().__init__(rayleigh_rv_quad, "rayleigh_rv")

    def get_model_impl(self, params_dict=None):
        return self.model_impl(_lambda=1.0)

    def get_params(self):
        return self.model_params


class UniformRVModel(BaseModel):
    def __init__(self):
        super().__init__(uniform_rv_quad, "uniform_rv")

    def get_model_impl(self, params_dict=None):
        return self.model_impl(_lambda=1.0)

    def get_params(self):
        return self.model_params


class BaseStaticKernel:
    def __init__(self):
        # assumed to be rbf kernel
        self.kernel_params = {
            "rbf_sigma": rbf_sigma
        }

    def get_params(self):
        return self.kernel_params

    def override_params(self, param_name, param_values):
        self.kernel_params[param_name] = param_values


class RBFStaticKernel(BaseStaticKernel):
    def get_kernel(self, params_dict):
        return sigkernel.sigkernel.RBFKernel(sigma=params_dict["rbf_sigma"])


class LaplaceStaticKernel(BaseStaticKernel):
    def get_kernel(self, params_dict):
        return sigkernel.sigkernel.LaplaceKernel(sigma=params_dict["rbf_sigma"])


class RFF_linearize(BaseStaticKernel):
    def __init__(self, data_shape):
        super().__init__()
        base_feature_count = np.prod(data_shape[1:])
        self.kernel_params["data_shape"] = [data_shape]
        self.kernel_params["rff_features"] = [
            np.min([5, base_feature_count]), np.min([10, base_feature_count]), np.min([25, base_feature_count])
        ]
        self.kernel_params["rff_metric"] = rff_metric
        self.kernel_params["rff_use_offset"] = rff_use_offset

    def get_kernel(self, params_dict):
        data_shape = self.kernel_params["data_shape"]
        return sigkernel.sigkernel.RFFKernel_1(
            rff_features=params_dict["rff_features"], sigma=params_dict["rbf_sigma"],
            metric=params_dict["rff_metric"], use_offset=params_dict["rff_use_offset"],
            data_shape=data_shape, linearize_data=True
        )


class RFF_standard(BaseStaticKernel):
    def __init__(self, data_shape):
        super().__init__()
        self.kernel_params["data_shape"] = [data_shape]
        self.kernel_params["rff_features"] = list({
            np.ceil(np.sqrt(data_shape[-1])),
            np.ceil(0.25 * data_shape[-1]),
            # np.ceil(0.5 * data_shape[-1]),
            # np.ceil(0.75 * data_shape[-1])
        })
        self.kernel_params["rff_metric"] = rff_metric
        self.kernel_params["rff_use_offset"] = rff_use_offset

    def get_kernel(self, params_dict):
        data_shape = self.kernel_params["data_shape"]
        return sigkernel.sigkernel.RFFKernel_1(
            rff_features=params_dict["rff_features"], sigma=params_dict["rbf_sigma"],
            metric=params_dict["rff_metric"], use_offset=params_dict["rff_use_offset"],
            data_shape=data_shape, linearize_data=False
        )


class RFF_RBFKernel(BaseStaticKernel):
    def __init__(self, n_features):
        super().__init__()
        self.kernel_params["n_features"] = [n_features]
        # self.kernel_params["rff_features"] = rff_features
        self.kernel_params["rff_features"] = [10, 25, 50, 100]
        self.kernel_params["rff_metric"] = rff_metric

    def get_kernel(self, params_dict):
        # we need to keep the size of kernel obtained at construction, not from stored parameters
        # as train size will be in the stored params for test kernel
        _n_features = self.kernel_params["n_features"][0]
        return sigkernel.sigkernel.RFFKernel(
            dims=params_dict["rff_features"], metric=params_dict["rff_metric"],
            gamma=params_dict["rbf_sigma"], length=_n_features)
