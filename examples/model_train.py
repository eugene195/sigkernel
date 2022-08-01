import argparse
import copy
import pickle
import random
from collections import defaultdict
from time import sleep

import torch
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import sigkernel
from examples.data_loader import CustomDataLoader
from examples.global_config import DEFAULT_KERNEL_HYPERPARAMS, PDE_LAMBDAS, all_parameter_combinations, model_impls
from examples.utils import exit_after, set_all_seeds, ModelStorage


@exit_after(500)
def _inner_train(rff_features, rff_metric, sigma, pde_impl, x_train, svc_parameters, y_train):
    try:
        # define static kernel
        # static_kernel = sigkernel.sigkernel.RBFKernel(sigma=sigma)
        static_kernel = sigkernel.sigkernel.RFFKernel(dims=int(rff_features * x_train.size(dim=2)), metric=rff_metric,
                                                      gamma=sigma, length=x_train.size(dim=2))
        # static_kernel = sigkernel.sigkernel.TensorSketchKernel()

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


def update_best_score(db_storage, model_name, ds_name, params_dict, model_pkl):
    db_storage.insert_model_results(model_name=model_name, ds_name=ds_name,
                                    params_dict=params_dict, model_pkl=model_pkl)


def get_new_xy_train_ts(x_train_const, y_train_const):
    x_train = copy.deepcopy(x_train_const)
    y_train = copy.deepcopy(y_train_const)

    # ZG: scale below 1
    x_train /= x_train.max()

    # encode outputs as labels
    y_train = LabelEncoder().fit_transform(y_train)
    return x_train, y_train


def train(dataset, seed, db_storage):
    svc_parameters = DEFAULT_KERNEL_HYPERPARAMS
    x_train_const, y_train_const, x_test_const, y_test_const = dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test

    total_param_count = len(all_parameter_combinations)
    # start grid-search
    best_scores_train = defaultdict(float)
    for pde_impl_name, pde_impl_func in model_impls.items():

        model_name = 'signature pde {}'.format(pde_impl_name)
        db_storage.insert_model(model_name)
        model_progress = db_storage.find_model_progress(model_name, dataset.ds_name, not_found_raise=False)
        if model_progress:
            trained_model_params, _ = db_storage.find_model_results(model_name, dataset.ds_name)
            best_scores_train[model_name] = trained_model_params["_best_score"]

        for _p_count, params_set in enumerate(all_parameter_combinations):
            if (best_scores_train[model_name] != 1.0) and (model_progress < _p_count):
                # fixme +1 creates an off-by-one upon recovery
                db_storage.update_model_progress(model_name=model_name, ds_name=dataset.ds_name, iteration=_p_count + 1)
                _add_time, _ts_scale_factor, _add_ll, _rbf_sigma, _rff_metric, _rff_features = params_set

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

                    try:
                        svc_model = _inner_train(_rff_features, _rff_metric, _rbf_sigma, pde_impl_func(pde_scale),
                                                 x_train, svc_parameters, y_train)
                        if svc_model and svc_model.best_score_ > best_scores_train[model_name]:
                            _best_score = svc_model.best_score_
                            best_scores_train[model_name] = _best_score
                            svc_model_pkl = pickle.dumps(svc_model, pickle.HIGHEST_PROTOCOL)
                            local_vars = vars()
                            params_dict = {
                                x: local_vars[x]
                                for x in ("seed", "_add_time", "_add_ll", "_ts_scale_factor", "_rbf_sigma",
                                          "pde_scale", "_best_score", "_rff_metric", "_rff_features")
                            }
                            print()
                            print("Updating best score for model {}".format(model_name))
                            print("Old score: {} New score: {}".format(best_scores_train[model_name],
                                                                       svc_model.best_score_))
                            print("Parameters {}".format(params_dict))
                            print()
                            update_best_score(db_storage, model_name, dataset.ds_name, params_dict, svc_model_pkl)
                    except KeyboardInterrupt as error:
                        print("Iteration failed due to timeout")
                        raise
    for k, v in best_scores_train.items():
        print("model name: {}, score: {}".format(k, v))


def run(ds_name, dataset_pctg, seed=None):
    data_loader = CustomDataLoader(dataset_pctg=dataset_pctg)
    dataset = data_loader.load_ds(ds_name)

    if not seed:
        seed = random.randrange(2**32 - 1)
    set_all_seeds(seed)

    db_file_name = "./{}_{}.results".format(ds_name, seed)

    db_storage = ModelStorage(db_file_name)
    db_storage.create_model_table()
    db_storage.insert_dataset(ds_name)

    train(dataset, seed, db_storage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', help='Dataset name')
    parser.add_argument('--seed', type=int, help='Use fixed seed instead of a random one')
    parser.add_argument('--dataset_pctg', default=1.0, type=float, help='Fraction of available data to be used')
    args = parser.parse_args()
    run(args.ds_name, args.dataset_pctg, args.seed)
