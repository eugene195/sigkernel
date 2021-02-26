import numpy as np
import copy
import math
import iisignature
from scipy.ndimage.interpolation import shift
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import as_float_array

def transform(paths, at=False, ll=False, scale=1e-1):
    paths = scale*paths
    if ll:
        paths = LeadLag().fit_transform(paths)
    if at:
        paths = AddTime().fit_transform(paths)
    return np.array(paths)

def normalize(sigs, width, depth):
    new_sigs = []
    for sig in tqdm(sigs):
        new_sig = np.zeros_like(sig)
        for k in range(depth):
            dim = width*(width**(k)-1)
            new_sig[dim:dim + width**(k+1)] = math.factorial(k+1)*sig[dim:dim + width**(k+1)]
        new_sigs.append(new_sig)
    return new_sigs

class AddTime(BaseEstimator, TransformerMixin):
    def __init__(self, init_time=0., total_time=1.):
        self.init_time = init_time
        self.total_time = total_time

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        t = np.linspace(self.init_time, self.init_time + 1, len(X))
        return np.c_[t, X]

    def transform(self, X, y=None):
        return [self.transform_instance(x) for x in X]

class Reversion(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [as_float_array(x[::-1]) for x in X]


class LeadLag(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        lag = []
        lead = []

        for val_lag, val_lead in zip(X[:-1], X[1:]):
            lag.append(val_lag)
            lead.append(val_lag)

            lag.append(val_lag)
            lead.append(val_lead)

        lag.append(X[-1])
        lead.append(X[-1])

        return np.c_[lag, lead]

    def transform(self, X, y=None):
        return [self.transform_instance(x) for x in X]

class Dyadic(BaseEstimator, TransformerMixin):
    def __init__(self, depth):
        self.depth = depth

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        n_splits = 2**self.depth

        intervals = np.array_split(X, n_splits)

        for i in range(1, len(intervals)):
            intervals[i] = np.r_[[intervals[i - 1][-1]], intervals[i]]

        return [as_float_array(interval) for interval in intervals]

    def transform(self, X, y=None):
        return [self.transform_instance(x) for x in X]

class PenOff(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):

        # Add pen-off
        X_transformed = np.c_[X, np.ones(len(X))]
        last = np.array(copy.deepcopy(X_transformed[-1]))
        last[-1] = 0.

        X_transformed = np.r_[X_transformed, [last]]

        # Add home
        X_transformed = np.r_[np.zeros(X_transformed.shape[1]).reshape(1, -1),
                              X_transformed]

        return X_transformed

    def transform(self, X, Y=None):
        return [self.transform_instance(x) for x in X]

class Stroke_Augment(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        stroke = 0.
        output = []
        for c in X:
            output += [r + [stroke] for r in c]
            stroke += 1.
        return np.array(output)

    def transform(self, X, Y=None):
        return [self.transform_instance(x) for x in X]

class Ink_Augment(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        output = []
        ink = 0.
        for c in X:
            for d in c:
                output += d + [ink]
                if d != c[-1]:
                    ink += 1.
        return np.array(output).reshape(-1, 3)

    def transform(self, X, Y=None):
        return [self.transform_instance(x) for x in X]

class Pen_Augment(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        output = []
        for c in X:
            output += [c[0] + [1.]]
            output += [r + [0.] for r in c]
            output += [c[-1] + [1.]]
        return np.array(output[1:-1])

    def transform(self, X, Y=None):
        return [self.transform_instance(x) for x in X]


#===========================================================================================================
def white(steps, width, time=1.):
    mu, sigma = 0, math.sqrt(time / steps) 
    return np.random.normal(mu, sigma, (steps, width))

def brownian(steps, width, time=1.):
    path = np.zeros((steps + 1, width))
    np.cumsum(white(steps, width, time), axis=0, out=path[1:, :])
    return path

def brownian_perturbed(steps, width, time=1., amplitude=1.):
    path = brownian(steps, width, time)
    t = np.random.randint(steps)
    path[t:] = path[t:] + amplitude
    return path

def truncated_sig_kernel_naive(x,y,depth):
    sigx = iisignature.sig(x,depth,2)
    sigy = iisignature.sig(y,depth,2)
    k_true = np.ones((len(x),len(y)))
    for i,sx in enumerate(sigx):
        for j,sy in enumerate(sigy):
            k_true[i+1,j+1] = 1.+np.dot(sigx[i],sigy[j])
    return k_true

def truncated_sig_kernel(X, num_levels, order=-1, difference=True, sigma=1.):
    """
    Computes the (truncated) signature kernel matrix of a given array of sequences. 
    
    Inputs:
    :X: a numpy array of shape (num_seq, len_seq, num_feat) of num_seq sequences of equal length, len_seq, with num_feat coordinates
    :num_levels: the number of signature levels used
    :order: the order of the signature kernel as per Kiraly and Oberhauser, order=num_levels gives the full signature kernel, while order < num_levels gives a lower order approximation. Defaults to order=-1, which means order=num_levels
    :difference: whether to difference the time series before computations (defaults to True)
    :sigma: a scalar or an np array of shape (num_levels+1); a multiplicative factor for each level
    
    Output:
    :K: a numpy array of shape (num_seq, num_seq)
    """
    order = num_levels if order < 1 else order
    sigma = sigma * np.ones((num_levels + 1,), dtype=X.dtype)
    
    if difference:
        X = np.diff(X, axis=1)
    
    num_seq, len_seq, num_feat = X.shape
    
    M = np.reshape(X.reshape((-1, num_feat)) @ X.reshape((-1, num_feat)).T, (num_seq, len_seq, num_seq, len_seq))
    K = sigma[0] * np.ones((num_seq, num_seq), dtype=X.dtype) + sigma[1] * np.sum(M, axis=(1, 3))
    R = M[None, None, ...]
    
    for m in range(1, num_levels):
        d = min(m+1, order)
        R_next = np.empty((d, d, num_seq, len_seq, num_seq, len_seq), dtype=X.dtype)
        R_next[0, 0] = M * shift(np.cumsum(np.cumsum(np.sum(R, axis=(0, 1)), axis=1), axis=3), shift=(0, 1, 0, 1))
        for j in range(1, d):
            R_next[0, j] = 1./(j+1) * M * shift(np.cumsum(np.sum(R[:, j-1], axis=0), axis=1), shift=(0, 1, 0, 0))
            R_next[j, 0] = 1./(j+1) * M * shift(np.cumsum(np.sum(R[j-1, :], axis=0), axis=3), shift=(0, 0, 0, 1))
            for i in range(1, d):
                R_next[i, j] = 1./((j+1)*(i+1)) * M * R[i-1, j-1]
        R = R_next
        K += sigma[m+1] * np.sum(R, axis=(0, 1, 3, 5))
    return K