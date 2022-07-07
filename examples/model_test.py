import argparse
import copy
import pickle
from time import sleep

import torch
from sklearn.preprocessing import LabelEncoder

import sigkernel
from examples.data_loader import CustomDataLoader
from sigkernel.general_sig_functions import rayleigh_rv_quad, benchmark_finite_diff_impl, const_weight_kernel, \
    uniform_rv_quad

pde_impls = {
    "benchmark": benchmark_finite_diff_impl,
    "const": const_weight_kernel,
    # "exp": const_exp_kernel,
    "quad": rayleigh_rv_quad,
    "uniform": uniform_rv_quad
}


def test(dataset):
    # load trained models
    try:
        with open('../results/trained_models.pkl', 'rb') as file:
            trained_models = pickle.load(file)
    except:
        print('Models need to be trained first')

    # load final results from last run
    try:
        with open('../results/final_results.pkl', 'rb') as file:
            final_results = pickle.load(file)
    except:
        final_results = {}

    x_train_const, y_train_const, x_test_const, y_test_const = dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test

    for pde_impl_name, pde_impl_func in pde_impls.items():
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
        _add_time, _add_ll, _ts_scale_factor, _rbf_sigma, pde_scale, estimator, _ = trained_models[(dataset.ds_name, model_name)]

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

    # save results
    with open('../results/final_results.pkl', 'wb') as file:
        pickle.dump(final_results, file)


def run(ds_name, dataset_pctg):
    data_loader = CustomDataLoader(dataset_pctg=dataset_pctg)
    dataset = data_loader.load_ds(ds_name)
    test(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', help='Dataset name')
    parser.add_argument('--dataset_pctg', default=1.0, type=float, help='Fraction of available data to be used')
    args = parser.parse_args()
    run(args.ds_name, args.dataset_pctg)
