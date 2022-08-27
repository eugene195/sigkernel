import itertools

import numpy as np

from sigkernel.general_sig_functions import benchmark_finite_diff_impl, const_weight_kernel, rayleigh_rv_quad, \
    uniform_rv_quad

_allow_cuda = False

_kernels =  [
            'linear',
            'rbf',
            'gak',
            'truncated signature',
            'signature pde'
            ]

# define grid-search hyperparameters for SVC (common to all kernels)
DEFAULT_KERNEL_HYPERPARAMS = {'C': np.logspace(0, 4, 5), 'gamma': list(np.logspace(-4, 4, 9)) + ['auto']}
# DEFAULT_KERNEL_HYPERPARAMS = {'C': np.logspace(0, 4, 3), 'gamma': list(np.logspace(-4, 4, 5)) + ['auto']}


PDE_LAMBDAS = [1e-3, 7.5e-2, 1e-1, 0.5, 1., 5., 10.]


add_time_axis = [
    True,
    # False
]
rff_metric = [
    "rbf",
    # "laplace"
]

rff_features = [
    .25,
    .5,
    .75,
    .90
]

scale_transform = [
    .01, .1,
    1
]
add_lead_lag = [
    # True,
    False
]
# kernel_pde_scaling = [1e-3, 7.5e-2, 1e-1, 0.5, 1., 5., 10.]
rbf_sigma = [
    1e-3, 5e-3,
    1e-2,
    2.5e-2, 5e-2, 7.5e-2,
    1e-1,
    2.5e-1,
    5e-1,
    7.5e-1,
    1.,
    2., 5.,
    10.
]
# kernel_C = np.logspace(0, 4, 5)
# kernel_Gamma = list(np.logspace(-4, 4, 9)) + ['auto']

all_parameter_combinations = list(itertools.product(
    add_time_axis,
    scale_transform,
    add_lead_lag,
    # needs to be treated per-usecase
    # kernel_pde_scaling,
    rbf_sigma,
    # rff_metric,
    # rff_features,
))

model_impls = {
    # "benchmark": benchmark_finite_diff_impl,
    "const": const_weight_kernel,
    # "exp": const_exp_kernel,
    "quad": rayleigh_rv_quad,
    "uniform": uniform_rv_quad
}


_datasets = [
            # 'ArticularyWordRecognition',
            # 'BasicMotions', x
            # 'Cricket',
            # 'ERing', fixme: does not load
            # 'Libras', x
            # 'NATOPS', x
            # 'RacketSports', x
            # 'FingerMovements',
            # 'Heartbeat',
            # 'SelfRegulationSCP1',
            # 'UWaveGestureLibrary'
            ]
