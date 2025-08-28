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


def benchmark(train, test, coder, n, repeats, K):

    image_train, profile_train, name_train = train
    image_test, profile_test, name_test = test

    label_train = coder.transform(name_train)
    label_test = coder.transform(name_test)

    X_test = (image_test, profile_test)
    results = {}
    for run in range(repeats):
        idx = sample(label_train, n)
        X_train = np.concatenate((image_train[idx], profile_train[idx]))
        y_train = np.tile(label_train[idx], (2,))
        results[run] = {
            'pred': {},
            'true': coder.inverse_transform(label_test)
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
    # N = (2, 4, 8, 12, 16)
    # K = (1, 3, 5, 7, 9)
    N = (4, 8, 12, 16, 32, 64, 128, 256)
    K = (1, 3, 9, 15, 31, 51)
    REPEATS = 20
    results = {name: {} for name in embeddings.keys()}

    random.seed(0)
    np.random.seed(0)

    for name, data in tqdm(embeddings.items()):

        for fold in data.keys():

            results[name][fold] = {}

            train = (
                data[fold]['train']['image'],
                data[fold]['train']['profile'],
                data[fold]['train']['label'],
            )

            test = (
                data[fold]['test']['image'],
                data[fold]['test']['profile'],
                data[fold]['test']['label'],
            )

            for n in N:
                # k_vals = list(filter(lambda k: k < n, K))
                subresults = benchmark(train, test, coder, n, REPEATS, K)
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