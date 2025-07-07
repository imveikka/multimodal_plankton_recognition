"""
Preparation of multimodal CytoSense (Lab) data.
Author: Veikka Immonen
"""

from pathlib import Path
import polars as pl

if __name__ == '__main__':

    root = Path('.')
    annot_path = root / 'annotations.csv'

    df = (
            pl.scan_csv(
            root / 'Pulse-shapes_CS_images_FastVISION-plus_exp22.csv',
            schema_overrides={
                'FWS': pl.Float32, 'SWS': pl.Float32,
                'FL.Green': pl.Float32,  'FL.Yellow': pl.Float32, 
                'FL.Orange': pl.Float32, 'FL.Red': pl.Float32
            },
            null_values=['NA'],
            low_memory=True
        )
        .group_by('sp', 'file_id')
        .agg(pl.col('FWS', 'SWS', 'FL.Green', 'FL.Yellow', 'FL.Orange', 'FL.Red'))
        .sort('sp', 'file_id')
    ).collect()

    imgs = []
    profs = []
    classes = []

    for name, img_path, fws, sws, green, yellow, orange, red in df.rows():

        prof_path = root / name
        prof_path /= f'{img_path.replace('Cropped_With_Scalebar', 'Profile')}.csv'

        imgs.append(str(root / name / f'{img_path}.jpg'))
        profs.append(str(prof_path))
        classes.append(name)
        
        pl.DataFrame(
            {
               'FSC': fws,
               'SSC': sws,
               'Green': green,
               'Yellow': yellow,
               'Orange': orange,
               'Red': red
            }
        ).write_csv(root / prof_path)

    pl.DataFrame(
        {
            'image': imgs,
            'profile': profs,
            'class': classes
        }
    ).sort('class', 'image').write_csv(annot_path)
