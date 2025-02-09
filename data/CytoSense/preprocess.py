"""
Preparation of multimodal CytoSense data.

Author: Veikka Immonen

Given the raw annotation table (Pulse-shapes_annotated_CS_images.csv) and 
compressed image folders (or depending how you got the data, this is what I got)
of each plankton specie, this script preprocess the data into a form that can 
be handled more easier for model training and possible train/test splitting. 
Running this script following directories and files:

*   ./images/: contains of all images where nested directory structures are
    flattened. Each image is renamed by it's respective label (X) from the
    annotation table
*   ./others/: scatter and fluorence data of each sample, in which such data is
    available. Stored in csv-files with same naming format as in images.
*   /annotations.csv: New annotation table, where for each sample, the class
    name, and information of available modalities is tabulated.

Once you have all the data downloaded, the preprocessing can be done by

    $ python preprocess.py

Note that in order to re-run the script, previously created files and 
directories must be removed before:

    $ rm -rf images others annotations.csv
"""

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
        other = samples.iloc[:, -7:]
        if not other.iloc[0].isna().any():
            info['has_fluorence'] = True
            other.columns = ['FWS', 'SWS', 'Green', 'Yellow', 'Orange', 'Red', 'Curvature']
            other.to_csv(mod_path / f'{x}.csv', index=False)
        annotations = pd.concat([annotations, info], ignore_index=True)
    
    annotations.X = annotations.X.astype(int)
    annotations.has_image = annotations.has_image.astype(bool)
    annotations.has_fluorence = annotations.has_fluorence.astype(bool)
    annotations.to_csv('annotations.csv', index=False)

    # remove empty directories
    for path in filter(lambda x: x.is_dir(), img_path.rglob('*')):
        shutil.rmtree(path)




