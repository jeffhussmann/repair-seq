import logging
import os
import shutil
import subprocess
import sys

from pathlib import Path

import knock_knock.build_targets

import repair_seq.build_guide_specific_targets
import repair_seq.demux
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

def build_targets(base_dir, num_processes):
    SRA_sample_sheet = repair_seq.demux.load_SRA_sample_sheet()

    pairs_to_build = set(zip(SRA_sample_sheet['variable_guide_library'], SRA_sample_sheet['target_info_prefix']))

    for variable_guide_library, target_info_prefix in pairs_to_build:
        logging.info(f'Building guide-specific targets in {base_dir} for {target_info_prefix} {variable_guide_library}')

        original_target = target_info_prefix
        original_genbank_name = target_info_prefix

        repair_seq.build_guide_specific_targets.build_all_singles(base_dir,
                                                                  original_target,
                                                                  original_genbank_name,
                                                                  [variable_guide_library],
                                                                  num_processes=num_processes,
                                                                 )

def download_and_build_indices(base_dir, num_processes):
    for genome_name in ['hg19', 'bosTau7']:
        knock_knock.build_targets.download_genome_and_build_indices(base_dir, genome_name, num_threads=num_processes)

def download_SRA_data(base_dir, screen_name, debug=False):
    logging.info(f'Downloading data for {screen_name} into {base_dir}')

    data_dir = Path(base_dir) / 'data' / screen_name
    data_dir.mkdir(exist_ok=True, parents=True)

    SRR_accessions = repair_seq.demux.load_SRR_accessions().loc[screen_name].index

    fastq_dump_common_command = [
        'fastq-dump',
        '--split-3',
        '--origfmt',
        '--gzip',
        '--outdir', str(data_dir),
    ]

    if debug:
        fastq_dump_common_command.extend([
            '--maxSpotId', str(int(1e6)),
        ])
    
    for SRR_accession in SRR_accessions:
        fastq_dump_command = fastq_dump_common_command + [SRR_accession]
        subprocess.run(fastq_dump_command, check=True)