"""
Preparation of multimodal CytoSense data.
Author: Veikka Immonen
"""

from pathlib import Path
import glob
import polars as pl
import numpy as np

if __name__ == '__main__':

    root = Path('.')
    annot_path = root / 'annotations.csv'

    pattern = str(root / '**' / '*.jpg')
    image_paths = glob.glob(pattern, recursive=True)

    def fun(path):
        file = path.replace(str(root) + '/', '')
        return (
            file.rsplit('/', 1)[-1].replace('_Cropped_With_Scalebar', '')[:-4],
            {
                'class': file.split('/', 1)[0],
                'image': file
            }
        )

    lookup = dict(map(fun, image_paths))

    scan_args = dict(separator=' ',
                    schema_overrides={'FWS': pl.Float32, 'SWS': pl.Float32,
                                      'FL.Green': pl.Float32,  'FL.Yellow': pl.Float32, 
                                      'FL.Orange': pl.Float32, 'FL.Red': pl.Float32},
                    null_values=['NA'],
                    low_memory=True)

    df = (
        pl.concat((
            pl.scan_csv(root / 'PDexp_Micro_phyto_pulse-shapes.txt', **scan_args),
            pl.scan_csv(root / 'Uto_2020_pulse-shapes.txt', **scan_args)
        ))
            .drop_nulls()
            .filter((pl.col('ID') > 0), pl.concat_str(pl.col('Sample'), pl.col('ID'), separator='_').is_in(lookup))
            .group_by('ID', 'Sample')
            .agg(pl.col('FWS', 'SWS', 'FL.Green', 'FL.Yellow', 'FL.Orange', 'FL.Red'))
            .sort('ID', 'Sample')
    ).collect()


    imgs = []
    profs = []
    classes = []

    for id, sample, fws, sws, green, yellow, orange, red in df.rows():
        key = f'{sample}_{id}'
        img_path = lookup[key]['image']
        prof_path = img_path.replace('Cropped_With_Scalebar', 'Profile').replace('jpg', 'csv')
        imgs.append(img_path)
        profs.append(prof_path)
        classes.append(lookup[key]['class'])
        profile = pl.DataFrame(
            {
               'FSC': fws,
               'SSC': sws,
               'Green': green,
               'Yellow': yellow,
               'Orange': orange,
               'Red': red
            }
        )
        (
            profile
            .filter(*(pl.col(c) > 0 for c in profile.columns))
            .write_csv(root / prof_path)
        )

    pl.DataFrame(
        {
            'image': imgs,
            'profile': profs,
            'class': classes
        }
    ).sort('class', 'image').write_csv(annot_path)
