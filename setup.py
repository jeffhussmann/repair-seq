from setuptools import setup, Extension

setup(
    name='repair_seq',
    version='1.0.1',

    author='Jeff Hussmann',
    author_email='jeff.hussmann@gmail.com',
    url='https://github.com/jeffhussmann/repair-seq',
    description='processing, analysis, and visualization of data from Repair-seq screens',

    packages=[
        'repair_seq',
    ],

    scripts=[
        'repair_seq/repair-seq',
    ],

    python_requires='>=3.6',

    install_requires=[
        'hits==0.2.2',
        'knock_knock==0.3.2',
    ],

    setup_requires=['cython'],

    ext_package='repair_seq',
    ext_modules=[
        Extension('collapse_cython', ['repair_seq/collapse_cython.pyx']),
    ],
)
