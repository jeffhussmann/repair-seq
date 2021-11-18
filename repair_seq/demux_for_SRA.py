import argparse
import gzip
import itertools
import sys
from collections import defaultdict

from contextlib import ExitStack
from pathlib import Path

import pandas as pd
import tqdm
import yaml

import hits.utilities
import hits.fastq

parser = argparse.ArgumentParser()
parser.add_argument('base_dir', type=Path)
parser.add_argument('batch')
parser.add_argument('--has_UMIs', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--only_names', action='store_true')

args = parser.parse_args()

batch_dir = args.base_dir / 'data' / args.batch

systematic_name_fields = [
    'cell_type',
    'editing_modality',
    'protospacer_targeted',
    'programmed_edit',
    'guide_library',
    'replicate',
]

systematic_names = set()

# Report any sytematic_name_fields missing from any sample.
missing_fields = defaultdict(set)

if args.has_UMIs:
    pool_details = pd.read_csv(batch_dir / 'pool_details.csv',
                               dtype={'guide_library': str, 'replicate': str},
                               index_col='pool_name',
                              )
    # Samples that had multiple indices are given as a semi-colon separated list.
    pool_details['index'] = [s.split(';') for s in pool_details['index']]

    sample_sheet_fn = batch_dir / 'sample_sheet.yaml'
    pool_sample_sheet = yaml.safe_load(sample_sheet_fn.read_text())

    systematic_name_to_index = {}

    for pool_name, row in pool_details.iterrows():
        for name_field in systematic_name_fields:
            if name_field not in row or pd.isna(row[name_field]):
                missing_fields[pool_name].add(name_field)

        if len(missing_fields[pool_name]) == 0:
            systematic_name = '_'.join(map(str, row[systematic_name_fields]))
            systematic_names.add(systematic_name)

            # If systematic_name has been manually added back to sample sheet,
            # confirm that it is consistent.
            if 'systematic_name' in row:
                if row['systematic_name'] != systematic_name:
                    raise ValueError(row['systematic_name'], systematic_name)

            systematic_name_to_index[systematic_name] = row['index']

            if args.only_names:
                full_pool_name = f'{pool_sample_sheet["group_name"]}_{pool_name}'
                print(full_pool_name, systematic_name)

        else:
            print(f'Warning: {pool_name} is missing fields.')
            for name_field in missing_fields[pool_name]:
                print(f'\t{name_field}')

    sample_resolver = hits.utilities.get_one_mismatch_resolver(systematic_name_to_index).get

    def quartet_to_systematic_name(quartet):
        possible_samples = sample_resolver(quartet.I2.seq, {'unknown'})
        
        if len(possible_samples) == 1:
            sample = next(iter(possible_samples))
        else:
            sample = 'unknown'
        
        return sample

else:
    pool_details = pd.read_csv(batch_dir / 'gDNA_pool_details.csv',
                               dtype={'guide_library': str, 'replicate': str},
                               index_col='pool_name',
                              )
    sample_sheet_fn = batch_dir / 'gDNA_sample_sheet.yaml'
    pool_sample_sheet = yaml.safe_load(sample_sheet_fn.read_text())

    systematic_name_to_I7 = {}
    systematic_name_to_I5 = {}

    for _, row in pool_details.iterrows():
        if not row[systematic_name_fields].isna().any():
            systematic_name = '_'.join(map(str, row[systematic_name_fields]))
            systematic_names.add(systematic_name)
            systematic_name_to_I7[systematic_name] = row['I7_index']
            systematic_name_to_I5[systematic_name] = row['I5_index']

    for pool_name, row in pool_details.iterrows():
        for name_field in systematic_name_fields:
            if name_field not in row or pd.isna(row[name_field]):
                missing_fields[pool_name].add(name_field)

        if len(missing_fields[pool_name]) == 0:
            systematic_name = '_'.join(map(str, row[systematic_name_fields]))
            systematic_names.add(systematic_name)

            # If systematic_name has been manually added back to sample sheet,
            # confirm that it is consistent.
            if 'systematic_name' in row:
                if row['systematic_name'] != systematic_name:
                    raise ValueError(row['systematic_name'], systematic_name)

            systematic_names.add(systematic_name)
            systematic_name_to_I7[systematic_name] = row['I7_index']
            systematic_name_to_I5[systematic_name] = row['I5_index']

            if args.only_names:
                full_pool_name = f'{pool_sample_sheet["group_name"]}_{pool_name}'
                print(full_pool_name, systematic_name)

        else:
            print(f'Warning: {pool_name} is missing fields.')
            for name_field in missing_fields[pool_name]:
                print(f'\t{name_field}')

    I7_resolver = hits.utilities.get_one_mismatch_resolver(systematic_name_to_I7).get
    I5_resolver = hits.utilities.get_one_mismatch_resolver(systematic_name_to_I5).get

    def quartet_to_systematic_name(quartet):
        I7_samples = I7_resolver(quartet.I1.seq, {'unknown'})
        I5_samples = I5_resolver(quartet.I2.seq, {'unknown'})

        consistent_with_both = I7_samples & I5_samples

        if len(consistent_with_both) == 1:
            sample = next(iter(consistent_with_both))
        else:
            sample = 'unknown'
        
        return sample

if args.only_names:
    sys.exit()

all_lane_info = yaml.safe_load(sample_sheet_fn.read_text())['quartets']

demux_dir = args.base_dir / 'GEO_upload'
demux_dir.mkdir(exist_ok=True)

with ExitStack() as stack:
    demuxed_fhs = {}
    
    for systematic_name in systematic_names:
        for which in ['R1', 'R2']:
            fn = demux_dir / f'{systematic_name}_{which}.fastq.gz'
            fh = stack.enter_context(gzip.open(fn, 'wt'))
            demuxed_fhs[systematic_name, which] = fh

    for lane, lane_info in all_lane_info.items():
        fastq_fns = [batch_dir / (lane_info[which] + '.fastq.gz') for which in hits.fastq.quartet_order]
        quartets = hits.fastq.read_quartets(fastq_fns, standardize_names=True)

        # In debug mode, only process the first 1e5 reads.
        if args.debug:
            quartets = itertools.islice(quartets, 100000)

        for quartet in tqdm.tqdm(quartets, total=lane_info['num_reads']):
            sample = quartet_to_systematic_name(quartet)
            
            if sample != 'unknown':
                if args.has_UMIs:
                    # Store UMI sequence and quality scores in read names.
                    new_qname = f'{quartet.R1.name}_{quartet.I1.seq}_{quartet.I1.qual}'

                    quartet.R1.name = new_qname
                    quartet.R2.name = new_qname

                demuxed_fhs[sample, 'R1'].write(str(quartet.R1))
                demuxed_fhs[sample, 'R2'].write(str(quartet.R2))