import numpy as np
from tslearn.datasets import UCR_UEA_datasets
from sktime.datasets import load_UCR_UEA_dataset

class CustomDataLoader:
    def __init__(self, dataset_pctg=1.0, use_cache=True):
        self.use_cache = use_cache
        self.dataset_pctg = dataset_pctg
        self.datasets = [
            'ArticularyWordRecognition',
            # 'BasicMotions',
            'Cricket',
            # 'ERing', fixme: does not load
            # 'Libras',  # dataset pctg = 1.0
            # 'NATOPS',  # dataset pctg = 1.0
            # 'RacketSports',  # dataset pctg = 1.0
            'FingerMovements',
            'Heartbeat',
            'SelfRegulationSCP1',
            'UWaveGestureLibrary'
            ''
        ]

    def get_all_ds(self):
        return self.datasets

    def load_ds(self, ds_name):
        if ds_name != "ERing":
            x_train, y_train, x_test, y_test = UCR_UEA_datasets(use_cache=self.use_cache).load_dataset(ds_name)
        else:
            X, Y = load_UCR_UEA_dataset(name=ds_name)
            x_train, y_train, x_test, y_test = X[:30], Y[:30], X[30:300], Y[30:300]
            x_train = np.array([
                np.array([ts for ts in d]).T
                for i, d in x_train.iterrows()
            ])
            x_test = np.array([
                np.array([ts for ts in d]).T
                for i, d in x_test.iterrows()
            ])
        x_train = x_train[:int(len(x_train) * self.dataset_pctg), :, :]
        y_train = y_train[:int(len(y_train) * self.dataset_pctg)]
        # x_test = x_test[:int(len(x_test) * self.dataset_pctg), :, :]
        # y_test = y_test[:int(len(y_test) * self.dataset_pctg)]
        x_test = x_test[:int(len(x_test)), :, :]
        y_test = y_test[:int(len(y_test))]

        print("x_train: {}, y_train: {}, x_test: {}, y_test: {}".format(
            x_train.shape, y_train.shape, x_test.shape, y_test.shape
        ))
        return DataSet((x_train, y_train, x_test, y_test), ds_name)


class DataSet:
    def __init__(self, ds_tuple, ds_name):
        self.ds_name = ds_name
        self.x_train, self.y_train, self.x_test, self.y_test = ds_tuple