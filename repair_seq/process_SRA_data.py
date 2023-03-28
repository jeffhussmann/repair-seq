import logging
import os
import shlex
import shutil
import subprocess
import sys

from pathlib import Path

import knock_knock.build_targets

import repair_seq.demux
import repair_seq.demux_gDNA
import repair_seq.pooled_screen

def install_metadata(base_dir):
    logging.info(f'Installing metadata into {base_dir}')

    metadata_dir = Path(os.path.realpath(__file__)).parent / 'metadata'
    subdirs_to_copy = ['targets', 'guides']
    for subdir in subdirs_to_copy:
        src = metadata_dir / subdir
        dest = Path(base_dir) / subdir

        if dest.exists():
            print(f'Can\'t install to {base_dir}, {dest} already exists')
            sys.exit(1)

        shutil.copytree(str(src), str(dest))

    logging.info(f'Metadata installed in {base_dir}')

def download_and_build_indices(base_dir, num_processes):
    for genome_name in ['hg19', 'bosTau7']:
        knock_knock.build_targets.download_genome_and_build_indices(base_dir, genome_name, num_threads=num_processes)

def download_data(base_dir, screen_name, debug=False):
    logging.info(f'Downloading data for {screen_name} into {base_dir}')

    data_dir = Path(base_dir) / 'data' / screen_name
    data_dir.mkdir(exist_ok=True, parents=True)

    SRR_accessions = repair_seq.demux.load_SRR_accessions().loc[screen_name].index

    for SRR_accession in SRR_accessions:

        # prefetch into the same directory the fastqs will end up to avoid
        # potential issues with tmp space limits 

        sra_fn = data_dir / f'{SRR_accession}.sra'
        prefetch_command = f'prefetch {SRR_accession} --output-file {sra_fn} --max-size {int(1e12)}'
        subprocess.run(shlex.split(prefetch_command), check=True)

        fastq_dump_command = f'fastq-dump --split-3 --origfmt --gzip --outdir {data_dir}'

        if debug:
            fastq_dump_command += f' --maxSpotId {int(1e6)}'
        
        fastq_dump_command += f' {sra_fn}'
        subprocess.run(shlex.split(fastq_dump_command), check=True)

def demux(base_dir, screen_name, debug=False):
    has_UMIs = repair_seq.demux.load_SRA_sample_sheet().loc[screen_name, 'has_UMIs']
    
    if has_UMIs:
        demux_module = repair_seq.demux
    else:
        demux_module = repair_seq.demux_gDNA

    demux_module.demux_group(base_dir, screen_name, from_SRA=True, debug=debug)
