import argparse
import itertools
import pickle
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import SVR

import sigkernel
from examples.global_config import DEFAULT_KERNEL_HYPERPARAMS, rbf_sigma
from examples.model_classes import BenchmarkModel, RBFStaticKernel, ConstScalingModel, RayleighRVModel, UniformRVModel

# Helper function that extract rolling windows of historical prices of size h and means of the next future f prices.
from examples.model_train import update_best_score
from examples.utils import set_all_seeds, ModelStorage


def GetWindow(x, h_window=30, f_window=10):
    rows = []
    cols_size = len(x.columns)
    # First window
    X = np.array(x.iloc[:h_window, ]).reshape(cols_size, -1)
    rows.append(np.array(x.iloc[:h_window, ]))
    # Append next window
    for i in range(1, len(x) - h_window + 1):
        x_i = np.array(x.iloc[i:i + h_window, ]).reshape(cols_size, -1)
        X = np.append(X, x_i, axis=0)
        rows.append(np.array(x.iloc[i:i + h_window, ]))

    # Cut the end that we can't use to predict future price
    rolling_window = (pd.DataFrame(X)).iloc[:-f_window, ]
    return rows[:-f_window]
    # return rolling_window


def GetNextMean(x, h_window=30, f_window=10):
    return pd.DataFrame((x.rolling(f_window).mean().iloc[h_window + f_window - 1:, ]))


def PlotResult(y_train, y_test, y_train_predict, y_test_predict, name):
    train_len = len(y_train)
    test_len = len(y_test)

    # Visualise
    fig, ax = plt.subplots(1, figsize=(12, 5))
    ax.plot(y_train_predict, color='red')

    ax.plot(range(train_len, train_len + test_len),
            y_test_predict,
            label='Predicted average price',
            color='red', linestyle='--')

    ax.plot(np.array((y_train).append(y_test)),
            label='Actual average price',
            color='green')

    ax.axvspan(len(y_train), len(y_train) + len(y_test),
               alpha=0.3, color='lightgrey')

    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc="best")
    plt.xlabel('Days')
    plt.ylabel('Bitcoin prices')
    plt.savefig('../pictures/bitcoin_prices_prediction_{}'.format(name))
    plt.show()


def run_alternative(seed=None, is_train=True):
    ds_name = 'BTS_7_feat'

    if not seed:
        seed = random.randrange(2**32 - 1)
    set_all_seeds(seed)
    db_file_name = "./{}_{}.results".format(ds_name, seed)
    db_storage = ModelStorage(db_file_name)
    db_storage.create_model_table()
    db_storage.insert_dataset(ds_name)

    sev_feat_ds = pd.read_csv("../btc_data-paper_datasets/reg_seven.csv", sep=',')
    X = sev_feat_ds.iloc[:, 0:sev_feat_ds.shape[1]-1]
    y = sev_feat_ds['priceUSD']

    # y = np.ravel(y)

    # use last h_window observations to predict mean over next f_window observations
    h_window = 36
    f_window = 2

    # next mean price
    y = GetNextMean(y, h_window=h_window, f_window=f_window)

    # normal window features
    X_window = GetWindow(X, h_window, f_window)
    X_window = [
        MinMaxScaler().fit_transform(w)
        for w in X_window
    ]
    # y = y * 1e-5

    X_window = torch.tensor(X_window, dtype=torch.float64)

    x_train, x_test, y_train, y_test = train_test_split(X_window, y, test_size=0.2, shuffle=False)
    X_train = torch.tensor(x_train, dtype=torch.float64, device='cpu')
    X_test = torch.tensor(x_test, dtype=torch.float64, device='cpu')

    for _model in [
        # BenchmarkModel(),
        # ConstScalingModel(),
        RayleighRVModel(), UniformRVModel()
    ]:
        _model.override_params("add_time_axis", [False])
        _model.override_params("scale_transform", [1])
        _model.override_params("add_lead_lag", [False])

        if is_train:
            train_model(ds_name, X_train, y_train, seed, _model, RBFStaticKernel(), db_storage)
        else:
            test_model(ds_name, X_train, X_test, y_train, y_test, _model, RBFStaticKernel(), db_storage)

def run(seed=None, is_train=True):
    ds_name = 'BTC_price'
    if not seed:
        seed = random.randrange(2**32 - 1)
    set_all_seeds(seed)

    db_file_name = "./{}_{}.results".format(ds_name, seed)

    db_storage = ModelStorage(db_file_name)
    db_storage.create_model_table()
    db_storage.insert_dataset(ds_name)

    BTC_price = pd.read_csv('../data/gemini_BTCUSD_day.csv', header=1)

    # drop the first column and reverse order
    BTC_price = BTC_price.iloc[1:, :]
    BTC_price = BTC_price.iloc[::-1]
    BTC_price['Date'] = pd.to_datetime(BTC_price['Date'])
    BTC_price.set_index('Date', inplace=True)

    # select duration
    initial_date = '2017-06-01'
    finish_date = '2018-08-01'
    BTC_price = BTC_price.loc[BTC_price.index >= initial_date]
    BTC_price = BTC_price.loc[BTC_price.index <= finish_date]

    # %%

    # use only close price
    close_price = BTC_price.loc[:, 'Close']
    # close_price = TimeSeriesScalerMeanVariance().fit_transform(close_price.values[None,:])
    close_price = pd.DataFrame(np.squeeze(close_price))

    # use last h_window observations to predict mean over next f_window observations
    h_window = 36
    f_window = 2

    # next mean price
    y = GetNextMean(close_price, h_window=h_window, f_window=f_window)

    # normal window features
    # X_window = GetWindow(close_price, h_window, f_window).values
    X_window = GetWindow(close_price, h_window, f_window)
    X_window = sigkernel.transform(np.asarray(X_window), at=True, ll=True, scale=1e-5)
    X_window = torch.tensor(X_window, dtype=torch.float64)

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(X_window, y, test_size=0.2, shuffle=False)
    x_train = torch.tensor(x_train, dtype=torch.float64, device='cpu')
    x_test = torch.tensor(x_test, dtype=torch.float64, device='cpu')
    for _model in [
        # BenchmarkModel(),
        ConstScalingModel(),
        # RayleighRVModel(),
        # UniformRVModel()
    ]:
        _model.override_params("add_time_axis", [True])
        _model.override_params("scale_transform", [1e-5])
        _model.override_params("add_lead_lag", [True])

        if is_train:
            train_model(ds_name, x_train, y_train, seed, _model, RBFStaticKernel(), db_storage)
        else:
            test_model(ds_name, x_train, x_test, y_train, y_test, _model, RBFStaticKernel(), db_storage)


def train_model(ds_name, x_train, y_train, seed, model, static_kernel, db_storage):
    best_scores_train = defaultdict(lambda: 1e8)
    _model_name = 'signature pde {}'.format(model.model_name)
    db_storage.insert_model(_model_name)

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
            # Initialize the corresponding signature kernel
            signature_kernel = sigkernel.SigKernel(model.get_model_impl(params_dict),
                                                   static_kernel.get_kernel(params_dict),
                                                   dyadic_order=0, _naive_solver=True)
            # Gram matrix train
            G_train = signature_kernel.compute_Gram(x_train, x_train, sym=True).numpy()

            # fit the model
            svr = SVR(kernel='precomputed')
            svr_pde = GridSearchCV(estimator=svr, param_grid=DEFAULT_KERNEL_HYPERPARAMS, cv=5, n_jobs=-1)
            svr_pde.fit(G_train, np.squeeze(y_train))

            print(np.abs(1. - svr_pde.best_score_))
            if np.abs(1. - svr_pde.best_score_) < np.abs(1. - best_scores_train[_model_name]):
                _best_score = svr_pde.best_score_
                best_scores_train[_model_name] = _best_score
                svr_model_pkl = pickle.dumps(svr_pde, pickle.HIGHEST_PROTOCOL)
                print()
                print("Updating best score for model {}".format(_model_name))
                print("Old score: {} New score: {}".format(best_scores_train[_model_name],
                                                           svr_pde.best_score_))
                print("Parameters {}".format(params_dict))
                print()
                update_best_score(db_storage, _model_name, ds_name, params_dict, svr_model_pkl)
            db_storage.update_model_progress(model_name=_model_name, ds_name=ds_name, iteration=p_count)


def test_model(ds_name, x_train, x_test, y_train, y_test, model, static_kernel, db_storage):
    model_name = 'signature pde {}'.format(model.model_name)
    params_dict, fitted_model_instance = db_storage.find_model_results(model_name, ds_name)

    # Initialize the corresponding signature kernel
    signature_kernel = sigkernel.sigkernel.SigKernel(
        model.get_model_impl(params_dict),  static_kernel.get_kernel(params_dict),
        dyadic_order=0, _naive_solver=True
    )

    # Gram matrix test
    G_train = signature_kernel.compute_Gram(x_train, x_train, sym=True).numpy()
    G_test = signature_kernel.compute_Gram(x_test, x_train, sym=False).numpy()

    # predict
    y_train_predict = fitted_model_instance.predict(G_train)
    y_test_predict = fitted_model_instance.predict(G_test)

    # calculate errors
    p_error_test = mean_absolute_percentage_error(np.array(y_test).reshape(-1, 1),
                                                  np.array(y_test_predict).reshape(-1, 1))
    print(model_name, f'testing accuracy: {p_error_test} %')
    PlotResult(y_train, y_test, y_train_predict, y_test_predict, model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--seed', type=int, help='Use fixed seed instead of a random one')
    args = parser.parse_args()
    run(args.seed, is_train=args.train)
    # run_alternative(args.seed, is_train=args.train)
