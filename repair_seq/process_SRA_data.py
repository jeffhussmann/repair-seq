import argparse
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
            print(f'Can\'t install to {args.project_directory}, {dest} already exists')
            sys.exit(1)

        shutil.copytree(str(src), str(dest))

    logging.info(f'Metadata installed in {base_dir}')

def build_targets(base_dir, num_processes):

    logging.info(f'Building guide-specific targets in {base_dir}')

    SRA_sample_sheet = repair_seq.demux.load_SRA_sample_sheet()

    guide_library_names = SRA_sample_sheet['variable_guide_library'].unique()

    target_info_prefixes = SRA_sample_sheet['target_info_prefix'].unique()
    if len(target_info_prefixes) != 1:
        raise ValueError(target_info_prefixes)

    target_info_prefix = target_info_prefixes[0]

    original_target = target_info_prefix
    original_genbank_name = target_info_prefix

    repair_seq.build_guide_specific_targets.build_all_singles(base_dir,
                                                              original_target,
                                                              original_genbank_name,
                                                              guide_library_names,
                                                              num_processes=num_processes,
                                                             )

def download_and_build_indices(base_dir, num_processes):
    for genome_name in ['hg19', 'bosTau7']:
        knock_knock.build_targets.download_genome_and_build_indices(base_dir, genome_name, num_threads=num_processes)

def download_SRA_data(base_dir, screen_name, debug=False):
    logging.info(f'Downloading data for {screen_name} into {base_dir}')

    SRA_sample_sheet = repair_seq.demux.load_SRA_sample_sheet()

    data_dir = Path(base_dir) / 'data' / screen_name
    data_dir.mkdir(exist_ok=True)
    
    fastq_dump_common_command = [
        'fastq-dump',
        '--split-3',
        '--origfmt',
        '--gzip',
        '--outdir', str(data_dir),
    ]
    
    fastq_dump_command = fastq_dump_common_command + [SRA_sample_sheet.loc[screen_name, 'SRR_accession']]
    subprocess.run(fastq_dump_command, check=True)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s: %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                       )

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='subcommand', title='subcommands')
    subparsers.required = True

    def add_base_dir_arg(parser):
        parser.add_argument('base_dir',
                            type=Path,
                            help='The base directory to store data, metadata, and output.',
                           )

    def add_num_processes_arg(parser):
        available_CPUs = len(os.sched_getaffinity(0))
        parser.add_argument('--num_processes',
                            type=int,
                            help='Number of processors to use.',
                            default=min(8, available_CPUs),
                           )

    def add_screen_name_arg(parser):
        valid_screen_names = sorted(repair_seq.demux.load_SRA_sample_sheet().index)
        parser.add_argument('screen_name',
                            help='Name of screen.',
                            choices=valid_screen_names,
                            metavar='screen_name',
                           )

    def add_debug_arg(parser):
        parser.add_argument('--debug',
                            action='store_true',
                            help='Run on a small fraction of the data for debugging purposes.',
                           )

    parser_initial_setup = subparsers.add_parser('initial_setup',
                                                 help='Perform intial setup of metadata and reference genomes.',
                                                )

    add_base_dir_arg(parser_initial_setup)
    add_num_processes_arg(parser_initial_setup)

    def initial_setup(args):
        install_metadata(args.base_dir)
        build_targets(args.base_dir, args.num_processes)
        download_and_build_indices(args.base_dir, args.num_processes)

    parser_initial_setup.set_defaults(func=initial_setup)

    parser_download = subparsers.add_parser('download',
                                            help='Download data for a screen from SRA.',
                                           )
    add_base_dir_arg(parser_download)
    add_screen_name_arg(parser_download)
    add_debug_arg(parser_download)

    def download(args):
        download_SRA_data(args.base_dir, args.screen_name, debug=args.debug)

    parser_download.set_defaults(func=download)

    parser_process = subparsers.add_parser('process',
                                           help='Processes data for a screen. Requires screen data to have been downloaded.',
                                          )
    add_base_dir_arg(parser_process)
    add_screen_name_arg(parser_process)
    add_num_processes_arg(parser_process)
    add_debug_arg(parser_process)

    def process(args):
        repair_seq.demux.demux_group(args.base_dir, args.screen_name, from_SRA=True, debug=args.debug)
        pool = repair_seq.pooled_screen.get_pool(args.base_dir, args.screen_name)
        pool.process(num_processes=args.num_processes)

    parser_process.set_defaults(func=process)

    args = parser.parse_args()
    args.func(args)