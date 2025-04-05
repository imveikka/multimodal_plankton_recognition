import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

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
        help="Annotation table name: [name]_[train/valid/test/short].csv."
    )

    parser.add_argument(
        "-t",
        "--trainsize",
        type=int,
        default=32,
        help="Number of samples of each class in train set."
    )

    parser.add_argument(
        "-v",
        "--validsize",
        type=int,
        default=16,
        help="Number of samples of each class in validation set."
    )

    parser.add_argument(
        "-m",
        "--minsize",
        type=int,
        default=64,
        help="Minimum size to not be omitted from the dataset."
    )

    args = parser.parse_args()

    data_dir = Path(f'.')
    image_dir = data_dir / 'images'
    profile_dir = data_dir / 'profiles'
    annotations = pd.read_csv(data_dir / 'annotations.csv')
    annotations = annotations[~(annotations.iloc[:, -2:] == False).all(axis=1)]

    mask = (annotations == False).any(axis=1)
    unimodal = annotations[mask]
    multimodal = annotations[~mask]
    class_names, counts = np.unique(multimodal.class_name, return_counts=True)
    
    train = pd.DataFrame(columns=multimodal.columns)
    test = pd.DataFrame(columns=multimodal.columns)
    valid = pd.DataFrame(columns=multimodal.columns)
    short = pd.DataFrame(columns=multimodal.columns)

    for name, count in zip(class_names, counts):

        if count < args.minsize:
            continue

        annot = multimodal[multimodal.class_name == name]
        train_annot, test_annot = train_test_split(annot, train_size=.5)

        if len(train_annot) < args.trainsize:
            temp = train_annot.sample(n=args.trainsize - len(train_annot))
            train_annot = pd.concat([train_annot, temp])
        elif len(train_annot) > args.trainsize:
            train_annot, temp = train_test_split(
                train_annot, 
                train_size=args.trainsize
            )
            test_annot = pd.concat([test_annot, temp])

        if len(test_annot) <= args.validsize:
            valid_annot = test_annot.copy()
        else:
            valid_annot = test_annot.sample(n=args.validsize)

        train = pd.concat([train, train_annot])
        test = pd.concat([test, test_annot])
        valid = pd.concat([valid, valid_annot])

        short_annot = unimodal[unimodal.class_name == name]
        short = pd.concat([short, short_annot])
    
    n = (counts >= args.minsize).sum()
    
    name = args.name
    annot_dir = data_dir / name

    if not annot_dir.exists():
        annot_dir.mkdir()

    train.to_csv(annot_dir / f'{name}_train.csv')
    test.to_csv(annot_dir /  f'{name}_test.csv')
    valid.to_csv(annot_dir / f'{name}_valid.csv')
    short.to_csv(annot_dir / f'{name}_short.csv')

    print(
        f'Dataset of {n} classes created to annotation\n' \
        + f'files {name}/{name}_[train/test/valid/short].csv.'
    )
