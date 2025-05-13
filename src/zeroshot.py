import numpy as np
from sklearn.utils.extmath import weighted_mode
from pynndescent import NNDescent


class ANNZeroShot:


    def __init__(self, X, y, **nndescent_args):
        self.y_ = y.copy()
        self.index = NNDescent(X , **nndescent_args) 
        self.index.prepare()


    def kneighbors(self, *X, **query_args):
        return tuple(self.index.query(x, **query_args) for x in X)
    

    def predict(self, *X, **query_args):
        neighbors = zip(*self.kneighbors(*X, **query_args))
        idx, dist = map(np.hstack, neighbors)
        weights = self._get_weights(dist)
        classes = self.y_[idx]
        predictions, _ = weighted_mode(classes, weights, axis=1)
        return predictions.ravel()


    def _get_weights(self, dist):
        with np.errstate(divide="ignore"):
           dist = 1.0 / dist       
        inf_mask = np.isinf(dist)
        inf_row = np.any(inf_mask, axis=1)
        dist[inf_row] = inf_mask[inf_row]
        return dist
