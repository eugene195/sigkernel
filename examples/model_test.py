import argparse
import copy
from time import sleep

import torch
from sklearn.preprocessing import LabelEncoder

import sigkernel
from examples.data_loader import CustomDataLoader
from examples.global_config import model_impls
from examples.utils import ModelStorage, set_all_seeds


def extract_model_params(db_storage, model_name, ds_name):
    trained_model_params = db_storage.find_model_results(model_name, ds_name)
    return (trained_model_params[k] for k in [
        "_add_time", "_add_ll", "_ts_scale_factor", "_rbf_sigma", "pde_scale", "estimator"
    ])

def test(dataset, db_storage):
    x_train_const, y_train_const, x_test_const, y_test_const = dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test

    for pde_impl_name, pde_impl_func in model_impls.items():
        print("Testing using : {}".format(pde_impl_name))
        model_name = 'signature pde {}'.format(pde_impl_name)

        x_train = copy.deepcopy(x_train_const)
        x_test = copy.deepcopy(x_test_const)
        y_train = copy.deepcopy(y_train_const)
        y_test = copy.deepcopy(y_test_const)

        x_train /= x_train.max()
        x_test /= x_test.max()

        # encode outputs as labels
        y_test = LabelEncoder().fit_transform(y_test)

        # extract information from training phase
        (_add_time, _add_ll, _ts_scale_factor,
        _rbf_sigma, pde_scale, estimator, _) = extract_model_params(db_storage,
            model_name, dataset.ds_name
        )

        # path-transform and subsampling
        x_train = sigkernel.transform(x_train, at=_add_time, ll=_add_ll, scale=_ts_scale_factor)
        x_test = sigkernel.transform(x_test, at=_add_time, ll=_add_ll, scale=_ts_scale_factor)

        device = 'cpu'
        dtype = torch.float64

        # numpy -> torch
        x_train = torch.tensor(x_train, dtype=dtype, device=device)
        x_test = torch.tensor(x_test, dtype=dtype, device=device)

        # define static kernel
        static_kernel = sigkernel.sigkernel.RBFKernel(sigma=_rbf_sigma)

        # initialize corresponding signature PDE kernel
        signature_kernel = sigkernel.sigkernel.SigKernel(
            pde_impl_func(pde_scale), static_kernel, dyadic_order=0, _naive_solver=True
        )

        # compute Gram matrix on test data
        # fixme?? x_test, x_train because we're using precomputed gram from x_train
        G_test = signature_kernel.compute_Gram(x_test, x_train, sym=False).cpu().numpy()

        # record scores
        train_score = estimator.best_score_
        test_score = estimator.score(G_test, y_test)
        final_results[(dataset.ds_name, model_name)] = {f'training accuracy: {train_score} %',
                                                        f'testing accuracy: {test_score} %'}

        # empty memory
        del G_test
        torch.cuda.empty_cache()

        sleep(0.5)
        print(dataset.ds_name, model_name, final_results[dataset.ds_name, model_name])

        print('\n')


def run(ds_name, dataset_pctg, seed):
    data_loader = CustomDataLoader(dataset_pctg=dataset_pctg)
    dataset = data_loader.load_ds(ds_name)

    set_all_seeds(seed)
    db_file_name = "./{}_{}.results".format(ds_name, seed)
    db_storage = ModelStorage(db_file_name)

    test(dataset, db_storage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', help='Dataset name')
    parser.add_argument('--seed', type=int, help='Use fixed seed instead of a random one')
    parser.add_argument('--dataset_pctg', default=1.0, type=float, help='Fraction of available data to be used')
    args = parser.parse_args()
    run(args.ds_name, args.dataset_pctg, args.seed)
