import zipfile
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import shutil


if __name__ == '__main__':

    root = Path('.')
    img_path = root / 'images'
    mod_path = root / 'others'
    annot_path = root / 'annotations.csv'

    assert not any(map(lambda x: x.exists(), [img_path, mod_path, annot_path])), \
    """
    Preprocess is already executed, please remove ./images/, ./others/ and 
    ./annotations.csv to proceed.
    """
    
    img_path.mkdir()
    mod_path.mkdir()
    
    archives = list(filter(lambda x: x.name.endswith('.zip'), root.iterdir()))
    table = pd.read_csv('./Pulse-shapes_annotated_CS_images.csv')
    annotations = pd.DataFrame({'X': [], 'class_name': [], 'has_image': [], 'has_fluorence': []})

    # unzip the archives
    for archive in tqdm(archives, desc='Unzipping images... ', ncols=80):
        with zipfile.ZipFile(archive, 'r') as zip_ref:
            zip_ref.extractall(img_path)

    # preprocess using the table
    for x in tqdm(table.X.unique(), desc='Processing the table... ', ncols=80):
        samples = table[table.X == x]
        file_id = samples.file_id.iloc[0]
        info = pd.DataFrame({
            'X': [x], 'class_name': [samples.classes.iloc[0]],
            'has_image': [False], 'has_fluorence': [False]
        })
        if (img_path / f'{file_id}.jpg').exists():
            info['has_image'] = True
            (img_path / f'{file_id}.jpg').rename(img_path / f'{x}.jpg')
        other = samples.iloc[:, -5:]
        if not other.iloc[0].isna().any():
            info['has_fluorence'] = True
            other.to_csv(mod_path / f'{x}.csv', index=False)
        annotations = pd.concat([annotations, info], ignore_index=True)
    
    annotations.X = annotations.X.astype(int)
    annotations.has_image = annotations.has_image.astype(bool)
    annotations.has_fluorence = annotations.has_fluorence.astype(bool)
    annotations.to_csv('annotations.csv', index=False)

    # remove empty directories
    for path in filter(lambda x: x.is_dir(), img_path.rglob('*')):
        shutil.rmtree(path)




