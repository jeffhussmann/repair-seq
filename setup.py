from setuptools import setup, Extension
from pathlib import Path

metadata_fns = []
metadata_dir = Path('repair_seq/metadata')
for fn in metadata_dir.rglob('*'):
    metadata_fns.append(str(fn.relative_to('repair_seq')))

setup(
    name='repair_seq',
    version='1.0.4',

    author='Jeff Hussmann',
    author_email='jeff.hussmann@gmail.com',
    url='https://github.com/jeffhussmann/repair-seq',
    description='processing, analysis, and visualization of data from Repair-seq screens',

    packages=[
        'repair_seq',
        'repair_seq/visualize',
        'repair_seq/test',
    ],

    package_data={
        'repair_seq': metadata_fns,
    },

    scripts=[
        'repair_seq/repair-seq',
    ],

    python_requires='>=3.7',

    install_requires=[
        'h5py>=3.1.0',
        'hdbscan==0.8.26',
        'hits>=0.3.3',
        'knock_knock>=0.3.8',
        'numba==0.51.2',
        'seaborn>=0.11.0',
        'umap-learn==0.4.6',
    ],

    setup_requires=['cython'],

    ext_package='repair_seq',
    ext_modules=[
        Extension('collapse_cython', ['repair_seq/collapse_cython.pyx']),
    ],
)
