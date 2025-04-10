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
*   ./profiles/: scatter and fluorescence data of each sample, in which such 
    data is available. Stored in csv-files with same naming format as used 
    with images.
*   /annotations.csv: Global annotation table, where for each sample, the class
    name, and information of available modalities is tabulated. Used to
    generate train, validation and test splits.

Download the data to zips/ directory. Samples of each class is stored in a
separate zip file. Leave them as they are. Once you have all the data 
downloaded, the preprocessing can be done by

    $ python preprocess.py

Note that in order to re-run the script, previously created files and 
directories must be removed before:

    $ rm -rf images profiles annotations.csv
"""

import zipfile
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import shutil
from multiprocessing import Pool

if __name__ == '__main__':

    root = Path('.')
    img_path = root / 'images'          # image data
    mod_path = root / 'profiles'        # optical profiles
    annot_path = root / 'annotations.csv'
    zips_path = root / 'zips'

    assert not any(map(lambda x: x.exists(), [img_path, mod_path, annot_path])), \
    """
    Preprocess is already executed, please remove ./images/, ./profiles/ and 
    ./annotations.csv to proceed.
    """
    
    img_path.mkdir()
    mod_path.mkdir()
    
    archives = list(filter(lambda x: x.name.endswith('.zip'), zips_path.iterdir()))
    table = pd.read_csv('./Pulse-shapes_annotated_CS_images.csv', low_memory=False)
    table = table.dropna()
    annotations = pd.DataFrame({'ID': [], 'class_name': []})

    # unzip the archives
    for archive in archives:
        with zipfile.ZipFile(archive, 'r') as zip_ref:
            zip_ref.extractall(img_path)

    def process(id, x, table=table):

        samples = table[table.X == x]
        file_id = samples.file_id.iloc[0]

        info = pd.DataFrame({
            'ID': [id], 'class_name': [samples.classes.iloc[0]],
        })

        (img_path / f'{file_id}.jpg').rename(img_path / f'{x}.jpg')

        profile = samples.iloc[:, -7:-1]
        profile = profile.loc[(profile != 0).all(axis=1)]
        profile.columns = ['FSC', 'SSC', 'Green', 'Yellow', 'Orange', 'Red']
        profile.to_csv(mod_path / f'{id}.csv', index=False)

        return info

    # preprocess using the table
    annotations = enumerate(table.X.unique(), 1)
    with Pool() as pool:
        annotations = pool.starmap(process, annotations)

    annotations = pd.concat(annotations, ignore_index=True)
    annotations.ID = annotations.ID.astype(int)
    annotations.to_csv('annotations.csv', index=False)

    # remove empty directories
    for path in filter(lambda x: x.is_dir(), img_path.rglob('*')):
        shutil.rmtree(path)

