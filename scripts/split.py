import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    pd.options.mode.chained_assignment = None  # default='warn'
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", 
        "--dataset",
        help="Dataset, location of annotation file."
    )

    parser.add_argument(
        "-s", 
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducibility"
    )

    parser.add_argument(
        "-n", 
        "--name",
        default="split",
        help="Annotation table name: [name]/[train/valid/test].csv."
    )

    parser.add_argument(
        "-t",
        "--trainsize",
        type=int,
        default=16,
        help="Number of samples of each class in train set."
    )

    parser.add_argument(
        "-m",
        "--minsize",
        type=int,
        default=32,
        help="Minimum size to not be omitted from the dataset."
    )

    args = parser.parse_args()

    data_dir = Path(args.dataset)
    annotations = pd.read_csv(data_dir / 'annotations.csv')

    class_names, counts = np.unique(annotations['class'], return_counts=True)
    
    train = pd.DataFrame(columns=annotations.columns)
    test = pd.DataFrame(columns=annotations.columns)

    for name, count in zip(class_names, counts):

        if count < args.minsize:
            continue

        annot = annotations[annotations['class'] == name]
        train_annot, test_annot = train_test_split(annot, train_size=args.trainsize)

        train = pd.concat([train, train_annot])
        test = pd.concat([test, test_annot])
    
    n = (counts >= args.minsize).sum()
    
    name = args.name
    annot_dir = data_dir / name

    if not annot_dir.exists():
        annot_dir.mkdir()

    stepback = name.count('/') + 1
    train_annot.loc[:, ['image', 'profile']] = train[['image', 'profile']].apply(lambda x: '../' * stepback + x)
    test_annot.loc[:, ['image', 'profile']] = test[['image', 'profile']].apply(lambda x: '../' * stepback + x)

    train.to_csv(annot_dir / f'train.csv')
    test.to_csv(annot_dir /  f'test.csv')

    print(
        f'Dataset of {n} classes created to annotation\n' \
        + f'files {data_dir}/{name}/[train/test].csv.'
    )
