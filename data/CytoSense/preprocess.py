"""
(Copy) Preparation of multimodal CytoSense data.

Author: Veikka Immonen

Given the raw annotation tables and image folders (depends how you got 
the data, this is what I got) of each plankton specie, this script preprocess 
the data into a form that can be handled more easier for model training and 
possible train/test splitting. Running this script following directories 
and files:

*   ./images/: contains of all images where nested directory structures are
    flattened. Each image is renamed by it's respective label (X) from the
    annotation table
*   ./profiles/: scatter and fluorescence data of each sample, in which such 
    data is available. Stored in csv-files with same naming format as used 
    with images.
*   /annotations.csv: Global annotation table, where for each sample, the class
    name, and information of available modalities is tabulated. Used to
    generate train, validation and test splits.

Once you have all the raw data downloaded, the preprocessing can be done by

    $ python preprocess.py

Note that in order to re-run the script, previously created files and 
directories must be removed before:

    $ rm -rf images profiles annotations.csv
"""

from pathlib import Path
import glob
import pandas as pd
import polars as pl
import shutil
from multiprocessing import Pool

if __name__ == '__main__':

    root = Path('.')
    img_path = root / 'images'          # image data
    mod_path = root / 'profiles'        # optical profiles
    annot_path = root / 'annotations.csv'
    orig_path = root / 'Annotated_CS_images'

    assert not any(map(lambda x: x.exists(), [img_path, mod_path, annot_path])), \
    """
    Preprocess is already executed, please remove ./images/, ./profiles/ and 
    ./annotations.csv to proceed.
    """
    
    # img_path.mkdir()
    # mod_path.mkdir()
    
    table = pd.read_csv(orig_path / 'Pulse-shapes_annotated_CS_images.csv', low_memory=False)
    annotations = pd.DataFrame({'ID': [], 'class_name': []})

    pattern = str(orig_path / '**' / '*.jpg')
    image_files = glob.glob(pattern, recursive=True)
    image_samples = [Path(file).name.split('_Cropped')[0] for file in image_files]

    data_dir = Path('./data/')
    
    scan = (
        pl.scan_csv(root / 'PDexp_Micro_phyto_pulse-shapes.txt', separator=' ')
        .head(1000000)
        .filter(pl.col('ID') != 0, pl.col('Sample').is_in(image_samples))
        .group_by('ID', maintain_order=True)
        .agg([pl.col('FWS'), pl.col('SWS'), pl.col('FL.Green'), pl.col('FL.Yellow'),
              pl.col('FL.Orange'), pl.col('FL.Red'),
              pl.col('Sample').first(), pl.col('cluster').first()])
    )
    
    df = scan.collect()

    print(df.shape)

    print(df['ID'].unique_counts())
    # def process(id, file_id, table=table):
    #     samples = table[table.file_id == file_id]
    #     file_id = samples.file_id.iloc[0]
    #     class_name = samples.sp.iloc[0]
    #     info = pd.DataFrame({
    #         'ID': [id], 'class_name': [class_name],
    #     })
    #     shutil.copy(orig_path / class_name / f'{file_id}.jpg',
    #                 img_path / f'{id}.jpg')
    #     profile = samples.iloc[:, -6:]
    #     profile = profile.loc[(profile != 0).all(axis=1)]
    #     profile.columns = ['FSC', 'SSC', 'Green', 'Yellow', 'Orange', 'Red']
    #     profile.to_csv(mod_path / f'{id}.csv', index=False)
    #     return info

    # # preprocess using the table
    # annotations = enumerate(table.file_id.unique(), 1)
    # with Pool() as pool:
    #     annotations = pool.starmap(process, annotations)

    # annotations = pd.concat(annotations, ignore_index=True)

    # annotations.ID = annotations.ID.astype(int)
    # annotations.to_csv('annotations.csv', index=False)

