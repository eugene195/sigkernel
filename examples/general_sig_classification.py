import argparse
import copy
import itertools
import pickle
import random
from collections import defaultdict

import torch
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import sigkernel
from examples.data_loader import CustomDataLoader
from examples.global_config import DEFAULT_KERNEL_HYPERPARAMS
from examples.model_classes import BenchmarkModel, ConstScalingModel, RayleighRVModel, UniformRVModel, RBFStaticKernel
from examples.model_train import update_best_score
from examples.utils import set_all_seeds, ModelStorage


def get_classifier_ts(x_const, y_const):
    x_ = copy.deepcopy(x_const)
    y_ = copy.deepcopy(y_const)

    # ZG: scale below 1
    x_ /= x_.max()

    # encode outputs as labels
    y_ = LabelEncoder().fit_transform(y_)
    return x_, y_


def train_model(ds_name, _x_train, y_train, seed, model, static_kernel, db_storage):
    best_scores_train = defaultdict(float)
    _model_name = 'signature pde {}'.format(model.model_name)
    db_storage.insert_model(_model_name)

    (x_train_const, y_train_const) = copy.deepcopy(_x_train), y_train

    model_progress = db_storage.find_model_progress(_model_name, ds_name, not_found_raise=False)
    if model_progress >= 0:
        trained_model_params, _est = db_storage.find_model_results(_model_name, ds_name)
        best_scores_train[_model_name] = _est.best_score_
    else:
        model_progress = -1
        db_storage.update_model_progress(model_name=_model_name, ds_name=ds_name, iteration=model_progress)

    params = {**model.get_params(), **static_kernel.get_params()}
    p_keys, p_values = zip(*params.items())
    bundles = list(itertools.product(*p_values))
    for p_count, bundle in enumerate(bundles):
        params_dict = dict(zip(p_keys, bundle))
        params_dict["seed"] = seed
        if (best_scores_train[_model_name] != 1.0) and (model_progress < p_count):
            print("Training model {} using parameters: {}".format(model.model_name, params_dict))
            print("Test {} of {}".format(p_count, len(list(bundles))))
            x_train = copy.deepcopy(x_train_const)
            x_train = sigkernel.transform(x_train,
                                          at=params_dict["add_time_axis"],
                                          ll=params_dict["add_lead_lag"],
                                          scale=params_dict["scale_transform"])
            x_train = torch.tensor(x_train, dtype=torch.float64, device='cpu')

            # initialize corresponding signature PDE kernel
            signature_kernel = sigkernel.sigkernel.SigKernel(model.get_model_impl(params_dict),
                                                             static_kernel.get_kernel(params_dict),
                                                             dyadic_order=0, _naive_solver=True)

            # compute Gram matrix on train data
            G_train = signature_kernel.compute_Gram(x_train, x_train, sym=True).cpu().numpy()

            # SVC sklearn estimator
            svc = SVC(kernel='precomputed', decision_function_shape='ovo')
            # fixme: cv=5??
            svc_model = GridSearchCV(estimator=svc, param_grid=DEFAULT_KERNEL_HYPERPARAMS, cv=5, n_jobs=-1)
            svc_model.fit(G_train, y_train_const)

            # empty memory
            del G_train
            print("Model score: {}".format(svc_model.best_score_))

            if svc_model and svc_model.best_score_ > best_scores_train[_model_name]:
                _best_score = svc_model.best_score_
                best_scores_train[_model_name] = _best_score
                svc_model_pkl = pickle.dumps(svc_model, pickle.HIGHEST_PROTOCOL)
                print()
                print("Updating best score for model {}".format(_model_name))
                print("Old score: {} New score: {}".format(best_scores_train[_model_name],
                                                           svc_model.best_score_))
                print("Parameters {}".format(params_dict))
                print()
                update_best_score(db_storage, _model_name, ds_name, params_dict, svc_model_pkl)

            db_storage.update_model_progress(model_name=_model_name, ds_name=ds_name, iteration=p_count)


def test_model(ds_name, x_test, x_train, y_test,  model, static_kernel, db_storage):
    print("Testing using : {}".format(model.model_name))
    model_name = 'signature pde {}'.format(model.model_name)
    final_results = {}

    # extract information from training phase
    params_dict, fitted_model_instance = db_storage.find_model_results(model_name, ds_name)

    # path-transform and subsampling
    x_train = sigkernel.transform(x_train, at=params_dict["add_time_axis"], ll=params_dict["add_lead_lag"],
                                  scale=params_dict["scale_transform"])
    x_test = sigkernel.transform(x_test, at=params_dict["add_time_axis"], ll=params_dict["add_lead_lag"],
                                 scale=params_dict["scale_transform"])

    x_train = torch.tensor(x_train, dtype=torch.float64, device='cpu')
    x_test = torch.tensor(x_test, dtype=torch.float64, device='cpu')

    # initialize corresponding signature PDE kernel
    signature_kernel = sigkernel.sigkernel.SigKernel(
        model.get_model_impl(params_dict), static_kernel.get_kernel(params_dict), dyadic_order=0, _naive_solver=True
    )

    # compute Gram matrix on test data
    # fixme?? x_test, x_train because we're using precomputed gram from x_train
    G_test = signature_kernel.compute_Gram(x_test, x_train, sym=False).cpu().numpy()

    # record scores
    train_score = fitted_model_instance.best_score_
    test_score = fitted_model_instance.score(G_test, y_test)
    final_results[(ds_name, model_name)] = {f'training accuracy: {train_score} %', f'testing accuracy: {test_score} %'}

    # empty memory
    del G_test
    print(ds_name, model_name, f'training accuracy: {train_score} %', f'testing accuracy: {test_score} %')
    print('\n')


def run(ds_name, dataset_pctg, seed=None, is_train=True):
    data_loader = CustomDataLoader(dataset_pctg=dataset_pctg)
    dataset = data_loader.load_ds(ds_name)

    if not seed:
        seed = random.randrange(2**32 - 1)
    set_all_seeds(seed)

    db_file_name = "./{}_{}.results".format(ds_name, seed)

    db_storage = ModelStorage(db_file_name)
    db_storage.create_model_table()
    db_storage.insert_dataset(ds_name)

    x_train, y_train = get_classifier_ts(dataset.x_train, dataset.y_train)
    x_test, y_test = get_classifier_ts(dataset.x_test, dataset.y_test)

    for _model in [BenchmarkModel(), ConstScalingModel(), RayleighRVModel(), UniformRVModel()]:
        if is_train:
            train_model(ds_name, x_train, y_train, seed,
                        model=_model, static_kernel=RBFStaticKernel(),
                        db_storage=db_storage)
        else:
            test_model(ds_name, x_test, x_train, y_test,
                       model=_model,  static_kernel=RBFStaticKernel(),
                       db_storage=db_storage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', help='Dataset name')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--seed', type=int, help='Use fixed seed instead of a random one')
    parser.add_argument('--dataset_pctg', default=1.0, type=float, help='Fraction of available data to be used')
    args = parser.parse_args()
    run(args.ds_name, args.dataset_pctg, args.seed, is_train=args.train)
