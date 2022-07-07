import argparse
import copy
import pickle
from collections import defaultdict
from time import sleep

import numpy as np
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import sigkernel
from examples.data_loader import CustomDataLoader
from examples.global_config import DEFAULT_KERNEL_HYPERPARAMS, PDE_LAMBDAS, all_parameter_combinations
from sigkernel.general_sig_functions import rayleigh_rv_quad, benchmark_finite_diff_impl, const_weight_kernel, \
    const_exp_kernel, uniform_rv_quad

pde_impls = {
    "benchmark": benchmark_finite_diff_impl,
    "const": const_weight_kernel,
    # "exp": const_exp_kernel,
    "quad": rayleigh_rv_quad,
    "uniform": uniform_rv_quad
}


def _inner_train(sigma, pde_impl, x_train, svc_parameters, y_train):
    try:
        # define static kernel
        static_kernel = sigkernel.sigkernel.RBFKernel(sigma=sigma)

        # initialize corresponding signature PDE kernel
        signature_kernel = sigkernel.sigkernel.SigKernel(pde_impl, static_kernel, dyadic_order=0, _naive_solver=True)

        # compute Gram matrix on train data
        G_train = signature_kernel.compute_Gram(x_train, x_train, sym=True).cpu().numpy()

        # SVC sklearn estimator
        svc = SVC(kernel='precomputed', decision_function_shape='ovo')
        svc_model = GridSearchCV(estimator=svc, param_grid=svc_parameters, cv=5, n_jobs=-1)
        svc_model.fit(G_train, y_train)

        # empty memory
        del G_train

        print("Model score: {}".format(svc_model.best_score_))

        torch.cuda.empty_cache()
        sleep(0.5)
        return svc_model
    except ValueError as e:
        print(e)
        return None


def update_best_score(model_name, ds_name, best_scores_train, trained_models, svc_model, params_tuple):
    print()
    print("Updating best score for model {}".format(model_name))
    print("Old score: {} New score: {}".format(best_scores_train[model_name], svc_model.best_score_))
    print("Parameters {}".format(params_tuple))
    print()
    best_scores_train[model_name] = svc_model.best_score_
    trained_models[(ds_name, model_name)] = params_tuple + (svc_model.best_score_,)


def get_new_xy_train_ts(x_train_const, y_train_const):
    x_train = copy.deepcopy(x_train_const)
    y_train = copy.deepcopy(y_train_const)

    # ZG: scale below 1
    x_train /= x_train.max()

    # encode outputs as labels
    y_train = LabelEncoder().fit_transform(y_train)
    return x_train, y_train


def train(dataset):
    # store best models in training phase
    try:
        with open('../results/trained_models.pkl', 'rb') as file:
            trained_models = pickle.load(file)
    except:
        trained_models = {}

    svc_parameters = DEFAULT_KERNEL_HYPERPARAMS
    x_train_const, y_train_const, x_test_const, y_test_const = dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test

    total_param_count = len(all_parameter_combinations)
    # start grid-search
    best_scores_train = defaultdict(float)
    for pde_impl_name, pde_impl_func in pde_impls.items():
        model_name = 'signature pde {}'.format(pde_impl_name)
        for _p_count, params_set in enumerate(all_parameter_combinations):
            if _p_count not in (29, 30, 31, 32, 33, 34, 35, 36) and (best_scores_train[model_name] != 1.0):

                _add_time, _ts_scale_factor, _add_ll, _rbf_sigma = params_set

                print("Training model {} using parameters: {}".format(pde_impl_name, params_set))
                print("Test {} of {}".format(_p_count, total_param_count))

                x_train, y_train = get_new_xy_train_ts(x_train_const, y_train_const)

                # path-transform
                # ZG: Paths interpolation to make them continuous
                # fixme: why scale?
                x_train = sigkernel.transform(
                    x_train, at=_add_time, ll=_add_ll, scale=_ts_scale_factor
                )

                device = 'cpu'
                dtype = torch.float64

                # numpy -> torch
                x_train = torch.tensor(x_train, dtype=dtype, device=device)

                kernel_pde_set = [1.] if pde_impl_name in ("benchmark", "quad", "uniform") else PDE_LAMBDAS
                for pde_scale in kernel_pde_set:
                    svc_model = _inner_train(_rbf_sigma, pde_impl_func(pde_scale), x_train, svc_parameters, y_train)
                    if svc_model and svc_model.best_score_ > best_scores_train[model_name]:
                        update_best_score(
                            model_name, dataset.ds_name, best_scores_train, trained_models, svc_model,
                            params_tuple=(_add_time, _add_ll, _ts_scale_factor, _rbf_sigma, pde_scale, svc_model)
                        )
    # save trained models
    with open('../results/trained_models.pkl', 'wb') as file:
        pickle.dump(trained_models, file)

    for k, v in best_scores_train.items():
        print("model name: {}, score: {}".format(k, v))


def run(ds_name, dataset_pctg):
    data_loader = CustomDataLoader(dataset_pctg=dataset_pctg)
    dataset = data_loader.load_ds(ds_name)
    train(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', help='Dataset name')
    parser.add_argument('--dataset_pctg', default=1.0, type=float, help='Fraction of available data to be used')
    args = parser.parse_args()
    run(args.ds_name, args.dataset_pctg)
