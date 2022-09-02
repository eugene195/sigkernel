import argparse
import copy
import gc
import itertools
import os
import pickle
import random
import sys
import warnings
from collections import defaultdict

import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils._testing import ignore_warnings

import sigkernel
from examples.data_loader import CustomDataLoader
from examples.global_config import DEFAULT_KERNEL_HYPERPARAMS
from examples.model_classes import BenchmarkModel, ConstScalingModel, RayleighRVModel, UniformRVModel, RBFStaticKernel, \
    RFF_RBFKernel, RFF_standard, RFF_linearize
from examples.model_train import update_best_score
from examples.utils import set_all_seeds, ModelStorage

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses


def get_classifier_ts(x_const, y_const):
    x_ = copy.deepcopy(x_const)
    y_ = copy.deepcopy(y_const)

    # ZG: scale below 1
    x_ /= x_.max()

    # encode outputs as labels
    y_ = LabelEncoder().fit_transform(y_)
    return x_, y_


@ignore_warnings(category=ConvergenceWarning)
def train_model(_x_train, y_train,
                _x_test, _y_test,
                _x_val, _y_val,
                seed, model, static_kernel):
    best_scores_train = defaultdict(float)
    best_scores_val = defaultdict(float)
    best_params = None
    _model_name = 'signature pde {}'.format(model.model_name)

    (
        x_train_const, x_test_const, x_val_const,
        y_train_const
    ) = copy.deepcopy(_x_train), copy.deepcopy(_x_test), copy.deepcopy(_x_val), y_train

    model_progress = -1
    params = {**model.get_params(), **static_kernel.get_params()}
    p_keys, p_values = zip(*params.items())
    bundles = list(itertools.product(*p_values))

    train_svc_model = GridSearchCV(
        estimator=SVC(kernel='precomputed',
                      decision_function_shape='ovo', max_iter=1e5),
        param_grid=DEFAULT_KERNEL_HYPERPARAMS, cv=5, n_jobs=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for p_count, bundle in enumerate(bundles):
            G_train, G_val, G_test = None, None, None

            params_dict = dict(zip(p_keys, bundle))
            params_dict["seed"] = seed
            if (best_scores_train[_model_name] != 1.0) and (model_progress < p_count):
                print("Training model {} using parameters: {}".format(model.model_name, params_dict))
                print("Test {} of {}".format(p_count, len(list(bundles))))

                def _get_x(X):
                    x_train = copy.deepcopy(X)
                    x_train = sigkernel.transform(x_train,
                                                  at=params_dict["add_time_axis"],
                                                  ll=params_dict["add_lead_lag"],
                                                  scale=params_dict["scale_transform"])
                    return torch.tensor(x_train, )

                x_train = _get_x(x_train_const)
                static_kernel.override_params("data_shape", x_train.shape)

                signature_kernel = sigkernel.sigkernel.SigKernel(
                    model.get_model_impl(params_dict),
                    static_kernel.get_kernel(params_dict),
                    dyadic_order=0, _naive_solver=True
                )

                G_train = signature_kernel.compute_Gram(x_train, x_train, sym=True).cpu().numpy()
                train_svc_model.fit(G_train, y_train_const)

                train_score = train_svc_model.best_score_
                print("Train iteration score: {}".format(train_score))

                if train_svc_model and train_score > best_scores_train[_model_name]:
                    x_val = _get_x(_x_val)
                    G_val = signature_kernel.compute_Gram(x_val, x_train, sym=False).cpu().numpy()

                    val_score = train_svc_model.score(G_val, _y_val)
                    if val_score > best_scores_val[_model_name]:
                        x_test = _get_x(_x_test)
                        G_test = signature_kernel.compute_Gram(x_test, x_train, sym=False).cpu().numpy()
                        test_score = train_svc_model.score(G_test, _y_test)

                        best_scores_train[_model_name] = train_score
                        best_scores_val[_model_name] = val_score
                        print()
                        print("Updating best score for model {}".format(_model_name))
                        print(f'Train score: {train_score}, val score: {val_score}, test score: {test_score}')
                        print("BEST_PARAMS: Parameters {}".format(params_dict))
                        best_params = params_dict
                del G_train
                del G_val
                del G_test
                gc.collect()
    print(f'Final results: \n'
          f'Train score: {best_scores_train[_model_name]}, '
          f'val score: {best_scores_val[_model_name]}, '
          f'test score: {test_score},'
          f'best parameters: {best_params}')


@ignore_warnings(category=ConvergenceWarning)
def run(ds_name, dataset_pctg, seed=None):
    data_loader = CustomDataLoader(dataset_pctg=dataset_pctg)
    dataset = data_loader.load_ds(ds_name, is_val=True)

    if not seed:
        seed = random.randrange(2**32 - 1)
    set_all_seeds(seed)

    x_train, y_train = get_classifier_ts(dataset.x_train, dataset.y_train)
    x_val, y_val = get_classifier_ts(dataset.x_val, dataset.y_val)
    x_test, y_test = get_classifier_ts(dataset.x_test, dataset.y_test)

    for _model in [
        BenchmarkModel(),
        # ConstScalingModel(),
        # RayleighRVModel(),
        # UniformRVModel()
    ]:
        for rff_kernel in [
            RFF_standard(data_shape=x_train.shape),
            # RFF_linearize(data_shape=x_train.shape)
        ]:
            train_model(
                x_train, y_train, x_test, y_test, x_val, y_val,
                seed, model=_model, static_kernel=rff_kernel,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', help='Dataset name')
    parser.add_argument('--seed', type=int, help='Use fixed seed instead of a random one')
    parser.add_argument('--dataset_pctg', default=1.0, type=float, help='Fraction of available data to be used')
    args = parser.parse_args()
    run(args.ds_name, args.dataset_pctg, args.seed)
