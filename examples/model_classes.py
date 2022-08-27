from examples.global_config import add_time_axis, scale_transform, add_lead_lag, PDE_LAMBDAS, rbf_sigma
from sigkernel.general_sig_functions import benchmark_finite_diff_impl, const_weight_kernel, rayleigh_rv_quad, \
    uniform_rv_quad
import sigkernel


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
        super().__init__(const_weight_kernel, "const")
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


class RBFStaticKernel:
    def __init__(self):
        pass

    @staticmethod
    def get_params():
        return {"rbf_sigma": rbf_sigma}

    @staticmethod
    def get_kernel(params_dict):
        return sigkernel.sigkernel.RBFKernel(sigma=params_dict["rbf_sigma"])
