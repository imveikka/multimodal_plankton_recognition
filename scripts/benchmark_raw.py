import numpy as np
import random
from tqdm import tqdm
import sys
import pickle
import argparse

from sklearn.preprocessing import LabelEncoder

sys.path.append('../')
from src.ann import ANNClassifier


def sample(y, n):
    idx = []
    orig = np.arange(len(y))
    for label in np.unique(y):
        i = list(orig[y == label])
        j = random.sample(i, n)
        idx.extend(j)
    return np.array(idx)


def benchmark(data, coder, n, repeats, K):
    images, profiles, names = data
    labels = coder.transform(names)
    length = len(labels)
    idx_full = set(range(length))
    results = {}
    for run in range(repeats):
        idx_train = sample(labels, n)
        idx_test = list(idx_full - set(idx_train))
        X_train = np.concatenate((images[idx_train], profiles[idx_train]))
        y_train = np.tile(labels[idx_train], (2,))
        X_test = (images[idx_test], profiles[idx_test])
        y_test = labels[idx_test]
        results[run] = {
            'pred': {},
            'true': coder.inverse_transform(y_test)
        }
        predictor = ANNClassifier(X_train, y_train, n_neighbors=32, 
                                  metric='euclidean', diversify_prob=0.0,
                                  pruning_degree_multiplier=3.0,
                                  low_memory=False,
                                  random_state=0)
        for k in K:
            pred = predictor.predict(*X_test, k=k, epsilon=0.3)
            results[run]['pred'][k] = coder.inverse_transform(pred)
    return results


def threshold(data, coder, th):
    images, profiles, names = data
    label = coder.transform(names)
    uniqs, counts = np.unique(label, return_counts=True)
    mask = counts >= th
    hits = tuple()
    for id in uniqs[mask]:
        hits += np.where(label == id)
    hits = np.concatenate(hits)
    return images[hits], profiles[hits], names[hits]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embeddings", help="Location to pickled embeddings.")
    parser.add_argument("-o", "--output", help="Path to pickled result data")
    args = parser.parse_args()

    with open(args.embeddings, 'rb') as buf:
        embeddings = pickle.load(buf)

    key_ = list(embeddings.keys())[0]
    coder = LabelEncoder().fit(embeddings[key_][1]['classes'])

    # Configure rest here
    N = (2, 4, 8, 12, 16)
    K = (1, 3, 5, 7, 9)

    N = (4, 8, 12, 16, 32, 64, 128, 256)
    K = (1, 3, 9, 15, 31, 51)

    TH = 20
    REPEATS = 20
    results = {name: {} for name in embeddings.keys()}

    random.seed(0)
    np.random.seed(0)

    for name, data in tqdm(embeddings.items()):

        for fold in data.keys():

            results[name][fold] = {}

            foo = data[fold]
            bar = (foo['image'], foo['profile'], foo['label'])
            bar = threshold(bar, coder, TH)

            for n in N:
                # k_vals = list(filter(lambda k: k < n, K))
                subresults = benchmark(bar, coder, n, REPEATS, K)
                results[name][fold][n] = subresults

    """
    Structure:

    results = {
        model: {
            fold: {
                n_per_class_in_gallery: subresults
                ...
            }
            ...
        } 
        ...
    }

    subresults = {
        simulation_id: {
            'true': array
            'pred': {
                k_neighbors_in_search: array
                ...
            }
        }
        ... 
    }
    """

    with open(args.output, 'wb') as buf:
        pickle.dump(results, buf)

if __name__ == '__main__':
    main()