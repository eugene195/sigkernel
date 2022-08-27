
from __future__ import print_function

import json
import os
import pickle
import random
import sqlite3
from sqlite3 import Error

import numpy as np
import torch
import sys
import threading
from time import sleep

try:
    import thread
except ImportError:
    import _thread as thread

try:  # use code that works the same in Python 2 and 3
    range, _print = xrange, print


    def print(*args, **kwargs):
        flush = kwargs.pop('flush', False)
        _print(*args, **kwargs)
        if flush:
            kwargs.get('file', sys.stdout).flush()
except NameError:
    pass


def cdquit(fn_name):
    # print to stderr, unbuffered in Python 2.
    print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush()  # Python 3 stderr is likely buffered.
    thread.interrupt_main()  # raises KeyboardInterrupt


def exit_after(s):
    '''
    use as decorator to exit process if
    function takes longer than s seconds
    '''

    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, cdquit, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer


def set_all_seeds(seed):
    random.seed(seed)
    # os.environ('PYTHONHASHSEED') = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


class ModelStorage:
    def __init__(self, ds_name):
        self.ds_name = ds_name
        self.conn = self.create_connection()

    def create_connection(self):
        """ create a database connection to a SQLite database """
        conn = None
        try:
            conn = sqlite3.connect(self.ds_name)
            print(sqlite3.version)
        except Error as e:
            print(e)
        return conn

    def create_model_table(self):
        c = self.conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS datasets (id integer PRIMARY KEY, name text NOT NULL UNIQUE)")
        c.execute("CREATE TABLE IF NOT EXISTS models (id integer PRIMARY KEY, name text NOT NULL UNIQUE)")
        c.execute("CREATE TABLE IF NOT EXISTS model_results ("
                  "id integer PRIMARY KEY, "
                  "model_params json, "
                  "dataset_id integer NOT NULL,"
                  "model_id integer NOT NULL,"
                  "model_pkl pickle, "
                  "FOREIGN KEY(dataset_id) REFERENCES datasets(id)"
                  "FOREIGN KEY(model_id) REFERENCES models(id)"
                  "CONSTRAINT unq UNIQUE (model_id, dataset_id)"
                  ")")
        c.execute("CREATE TABLE IF NOT EXISTS model_progress (id integer PRIMARY KEY, "
                  "iteration_nr integer NOT NULL,"
                  "model_id integer NOT NULL,"
                  "dataset_id integer NOT NULL,"
                  "FOREIGN KEY(model_id) REFERENCES model_results(id)"
                  "FOREIGN KEY(dataset_id) REFERENCES datasets(id)"
                  "CONSTRAINT unq UNIQUE (model_id, dataset_id)"
                  ")")

        self.conn.commit()

    def insert_dataset(self, ds_name):
        c = self.conn.cursor()
        c.execute("insert OR IGNORE into datasets (name) values (?)", (ds_name, ))
        self.conn.commit()

    def insert_model(self, model_name):
        c = self.conn.cursor()
        c.execute("insert OR IGNORE into models (name) values (?)", (model_name, ))
        self.conn.commit()

    def find_ds_by_name(self, ds_name):
        c = self.conn.cursor()
        c.execute("SELECT * FROM datasets WHERE name=?", (ds_name,))
        return c.fetchall()

    def find_model_by_name(self, model_name):
        c = self.conn.cursor()
        c.execute("SELECT * FROM models WHERE name=?", (model_name,))
        return c.fetchall()

    def find_model_results(self, model_name, ds_name, not_found_raise=True):
        model_id, _ = self.find_model_by_name(model_name)[0]
        dataset_id, _ = self.find_ds_by_name(ds_name)[0]

        c = self.conn.cursor()
        c.execute("SELECT * FROM model_results WHERE dataset_id=? AND model_id=?",
                  (dataset_id, model_id))
        records = c.fetchall()
        # model params are in pos 1
        res = None
        try:
            model_params = json.loads(records[0][1])
            model_pkl = pickle.loads(records[0][4])
            res = (model_params, model_pkl)
        except IndexError:
            if not_found_raise:
                raise
        return res

    def insert_model_results(self, model_name, ds_name, params_dict, model_pkl):
        c = self.conn.cursor()

        model_id, _ = self.find_model_by_name(model_name)[0]
        dataset_id, _ = self.find_ds_by_name(ds_name)[0]
        c.execute("insert OR REPLACE into model_results (dataset_id, model_id, model_params, model_pkl) "
                  "values (?, ?, ?, ?)",
                  (dataset_id, model_id, json.dumps(params_dict), sqlite3.Binary(model_pkl), ))
        self.conn.commit()

    def update_model_progress(self, model_name, ds_name, iteration):
        c = self.conn.cursor()

        model_id, _ = self.find_model_by_name(model_name)[0]
        dataset_id, _ = self.find_ds_by_name(ds_name)[0]

        c.execute("insert OR REPLACE into model_progress (iteration_nr, model_id, dataset_id) values (?, ?, ?)",
                  (iteration, model_id, dataset_id,))
        self.conn.commit()

    def find_model_progress(self, model_name, ds_name, not_found_raise=True):
        c = self.conn.cursor()

        model_id, _ = self.find_model_by_name(model_name)[0]
        dataset_id, _ = self.find_ds_by_name(ds_name)[0]

        c.execute("SELECT * FROM model_progress WHERE dataset_id=? AND model_id=?",
                  (dataset_id, model_id))
        records = c.fetchall()
        res = -1
        try:
            res = records[0][1]
        except IndexError:
            if not_found_raise:
                raise
        return res
