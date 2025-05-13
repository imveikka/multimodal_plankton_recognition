import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold


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
        "-k",
        "--kfolds",
        type=int,
        default=5,
        help="Number of folds. Must be at least 2.",
    )

    parser.add_argument(
        "-n", 
        "--name",
        default="split",
        help="Annotation table name: [name][k]/[train/test].csv."
    )

    args = parser.parse_args()

    data_dir = Path(args.dataset)
    annot = pd.read_csv(data_dir / 'annotations.csv')

    kfold = StratifiedKFold(n_splits=args.kfolds,
                            shuffle=True,
                            random_state=args.seed)

    for k, (train, test) in enumerate(kfold.split(annot, annot.class_name), 1):

        annot_dir = data_dir / f'{args.name}{k}'
        if not annot_dir.exists():
            annot_dir.mkdir()
        annot.iloc[train].to_csv(annot_dir / f'train.csv')
        annot.iloc[test].to_csv(annot_dir /  f'test.csv')

    print(
        f'Dataset folds created to annotation\n' \
        + f'files {args.name}[1-{args.kfolds}]/[train/test].csv.'
    )
