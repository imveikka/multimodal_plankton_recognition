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
    label_train = coder.transform(name_train)

    image_test, profile_test, name_test = test
    label_test = coder.transform(name_test)

    results = {}
    ann_kwargs = dict(
        n_neighbors=32, 
        metric='euclidean',
        diversify_prob=0.0,
        pruning_degree_multiplier=3.0,
        low_memory=False,
        random_state=0
    )
    for run in range(repeats):

        idx = sample(label_train, n)

        image_sample = image_train[idx]
        profile_sample = profile_train[idx]
        label_sample = label_train[idx]

        results[run] = {
            'pred': {k: {} for k in K},
            'true': coder.inverse_transform(label_test)
        }

        # image only
        predictor = ANNClassifier(image_sample, label_sample, **ann_kwargs)
        for k in K:
            results[run]['pred'][k] |= predict_k(
                predictor, coder,
                ('I - I', 'I - P', 'I - I+P'),
                ((image_test,), (profile_test,), (image_test, profile_test)),
                k=k, epsilon=.3
            )

        # profile only
        predictor = ANNClassifier(profile_sample, label_sample, **ann_kwargs)
        for k in K:
            results[run]['pred'][k] |= predict_k(
                predictor, coder,
                ('P - I', 'P - P', 'P - I+P'),
                ((image_test,), (profile_test,), (image_test, profile_test)),
                k=k, epsilon=.3
            )

        # image + profile
        data_train_double = np.concatenate((image_sample, profile_sample))
        label_train_double = np.tile(label_sample, (2,))
        predictor = ANNClassifier(data_train_double, label_train_double, **ann_kwargs)
        for k in K:
            results[run]['pred'][k] |= predict_k(
                predictor, coder,
                ('I+P - I', 'I+P - P'),
                ((image_test,), (profile_test,)),
                k=k, epsilon=.3
            )

    return results


def predict_k(predictor, coder, keys, X_list, **ann_kwargs):
    out = {}
    for key, X in zip(keys, X_list):
        pred = predictor.predict(*X, **ann_kwargs)
        out[key] = coder.inverse_transform(pred)
    return out


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

    # N = (4, 8, 12, 16, 32, 64, 128, 256)
    # K = (1, 3, 15, 31, 51)

    TH = 20
    REPEATS = 10
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
                k_neighbors_in_search: {
                    config: array 
                    ...
                }
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