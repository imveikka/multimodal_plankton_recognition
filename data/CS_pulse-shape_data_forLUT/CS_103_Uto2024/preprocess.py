"""
Preparation of multimodal CytoSense data.
Author: Veikka Immonen
"""

from glob import glob
import polars as pl
import os
from tqdm import tqdm

profile_files = sorted(glob('./*/*.csv'))
image_files = sorted(glob('./**/*.jpg', recursive=True))

def extract_img(file):
    keys = file.rsplit('/', 1)[-1][:-4].split(' ', 2)
    sample_type = keys[-1].split('_')
    return tuple(keys[:-1] + [sample_type[0], sample_type[-1]])

def extract_prof(file):
    new_path = file[:-4]
    os.makedirs(new_path, exist_ok=True)

    path, name = file.rsplit('/', 1)
    keys = name[:-4].split(' ', 2)
    sample_type = keys[-1].split('_', 1)
    key = tuple(keys[:-1] + [sample_type[0]])

    scan_args = dict(
        schema_overrides={'Particle ID': pl.Int64, 'FWS': pl.Float32,
                          'SWS': pl.Float32, 'FL Green': pl.Float32,
                          'FL Yellow': pl.Float32, 'Fl Orange': pl.Float32,
                          'FL Red': pl.Float32, 'Curvature': pl.Float32},
        null_values=['NA'],
    )
    df = (
            pl.scan_csv(file, **scan_args)
            .filter((pl.col('Particle ID') > 0))
            .group_by('Particle ID')
            .agg(pl.col('FWS', 'SWS', 'FL Green', 'FL Yellow', 'Fl Orange', 'FL Red'))
            .sort('Particle ID')
    ).collect()

    out = {}
    for id, fws, sws, green, yellow, orange, red in df.rows():
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
        dest = f'{new_path}/{key[0]} {key[1]} {key[2]}_Pulse_{id}.csv'
        (
            profile
            .filter(*(pl.col(c) > 0 for c in profile.columns))
            .write_csv(dest)
        )
        out |= {key + (str(id),): dest}

    return out

images = dict(zip(map(extract_img, image_files), image_files))
profiles = {}
for file in tqdm(profile_files):
    profiles |= extract_prof(file)

keys = sorted(set(images.keys()) & set(profiles.keys()))
pl.DataFrame(
    {
        'image': [images[key] for key in keys],
        'profile': [profiles[key] for key in keys],
        'class': ['unknown'] * len(keys)
    }
).write_csv('./annotations.csv')
