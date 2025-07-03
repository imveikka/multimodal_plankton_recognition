import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", 
        "--dataset",
        help="Dataset, location of annotation file and directories images/ and profiles/"
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

    data_dir = Path(args.dataset)
    image_dir = data_dir / 'images'
    profile_dir = data_dir / 'profiles'
    annotations = pd.read_csv(data_dir / 'annotations.csv')

    class_names, counts = np.unique(annotations.class_name, return_counts=True)
    
    train = pd.DataFrame(columns=annotations.columns)
    test = pd.DataFrame(columns=annotations.columns)
    valid = pd.DataFrame(columns=annotations.columns)

    for name, count in zip(class_names, counts):

        if count < args.minsize:
            continue

        annot = annotations[annotations.class_name == name]
        train_annot, test_annot = train_test_split(annot, train_size=.8)

        if len(train_annot) < args.trainsize:
            temp = train_annot.sample(n=args.trainsize-len(train_annot))
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
    
    n = (counts >= args.minsize).sum()
    
    name = args.name
    annot_dir = data_dir / name

    if not annot_dir.exists():
        annot_dir.mkdir()

    train.to_csv(annot_dir / f'train.csv')
    test.to_csv(annot_dir /  f'test.csv')
    valid.to_csv(annot_dir / f'valid.csv')

    print(
        f'Dataset of {n} classes created to annotation\n' \
        + f'files {name}/[train/test/valid].csv.'
    )
