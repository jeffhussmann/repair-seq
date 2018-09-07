#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path
from itertools import chain, islice
from collections import Counter

import pysam
import pandas as pd
import yaml
import tqdm; progress = tqdm.tqdm

from sequencing import mapping_tools, fastq

import collapse

class FastqQuartetSplitter(object):
    def __init__(self, base_path, reads_per_chunk=5000000):
        self.base_path = base_path
        self.reads_per_chunk = reads_per_chunk
        self.next_chunk_number = 0
        self.next_read_number = 0
        self.chunk_fhs = None
    
    def close(self):
        if self.chunk_fhs is not None:
            for fh in self.chunk_fhs:
                fh.close()
                
    def start_next_chunk(self):
        self.close()
  
        template = str(self.base_path)  + '/{}.{:05d}.fastq'
        fns = [template.format(which, self.next_chunk_number) for which in fastq.quartet_order]
        self.chunk_fhs = [open(fn, 'w') for fn in fns]
        
        self.next_chunk_number += 1
        
    def write(self, quartet):
        if self.next_read_number % self.reads_per_chunk == 0:
            self.start_next_chunk()
            
        for read, fh in zip(quartet, self.chunk_fhs):
            fh.write(str(read))
            
        self.next_read_number += 1

def hamming_distance(first, second):
    return sum(1 for f, s in zip(first, second) if f != s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--sample_dir', required=True)

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--demux_samples', action='store_true')
    mode_group.add_argument('--map_parallel', metavar='MAX_PROCS')
    mode_group.add_argument('--map_chunk', metavar='CHUNK')
    mode_group.add_argument('--demux_guides', action='store_true')

    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    sample_name = sample_dir.parts[-1]

    if args.demux_samples:
        raise ValueError
        sample_counts = Counter()

        sample_sheet = yaml.load((sample_dir / 'sample_sheet.yaml').read_text())
        sample_indices = list(sample_sheet['sample_indices'].items())
        group_name = sample_sheet['group_name']

        splitters = {}

        to_skip = set(sample_sheet['samples_to_skip'])

        def get_sample_input_dir(sample):
            return sample_dir.parent / '{}_{}'.format(group_name, sample) / 'input'

        for sample, index in sample_indices + [('unknown', '')]:
            if sample not in to_skip:
                base_path = get_sample_input_dir(sample)
                base_path.mkdir(parents=True, exist_ok=True)

                splitters[sample] = FastqQuartetSplitter(base_path)

        fastq_fns = [[sample_dir / name for name in sample_sheet[which]] for which in fastq.quartet_order]
        fn_quartets = (fastq.read_quartets(fns) for fns in zip(*fastq_fns))
        quartets = chain.from_iterable(fn_quartets)

        for quartet in progress(quartets, total=sample_sheet['num_reads']): 
            sample = 'unknown'
            
            for name, index in sample_indices:
                if hamming_distance(quartet.I2.seq, index) <= 1:
                    sample = name
            
            if sample not in to_skip:
                splitters[sample].write(quartet)

            sample_counts[sample] += 1
        
        for splitter in splitters.values():
            splitter.close()

        for sample, index in sample_indices + [('unknown', '')]:
            if sample not in to_skip:
                stats = {
                    'num_reads': sample_counts[sample],
                }
                stats_fn = get_sample_input_dir / 'stats.yaml'
                stats_fn.write_text(yaml.dump(stats, default_flow_style=False))

        pd.Series(sample_counts).to_csv(sample_dir / 'sample_counts.txt', sep='\t')

    elif args.map_parallel is not None:
        max_procs = args.map_parallel
        R1_fns = sorted((sample_dir / 'input').glob('R1.*.fastq'))
        chunks = [fn.suffixes[0].strip('.') for fn in R1_fns]

        subset_chunks = ['{:05d}'.format(i) for i in range(1)]
        chunks = [c for c in chunks if c in subset_chunks]

        parallel_command = [
            'parallel',
            '-n', '1', 
            '--verbose',
            '--max-procs', max_procs,
            './demultiplex.py',
            '--sample_dir', str(sample_dir),
            '--map_chunk', ':::',
        ] + chunks
        
        subprocess.run(parallel_command, check=True)

    elif args.map_chunk is not None:
        chunk = args.map_chunk

        R1_fn = sample_dir / 'input' / 'R1.{}.fastq'.format(chunk)
        STAR_index = '/home/jah/projects/britt/guides/STAR_index'
        output_dir = sample_dir / 'guide_mapping'
        output_dir.mkdir(exist_ok=True)
        output_prefix = output_dir / '{}.'.format(chunk)
        mapping_tools.map_STAR(R1_fn, STAR_index, output_prefix,
                               sort=False,
                               min_fraction_matching=0.9,
                               include_unmapped=True,
                               )

    elif args.demux_guides:
        stats = yaml.load((sample_dir / 'input' / 'stats.yaml').read_text())

        guide_dir = sample_dir / 'by_guide'
        guide_dir.mkdir(exist_ok=True)

        def get_fastq_fns(which):
            pattern = '{}*.fastq'.format(which)
            return sorted((sample_dir / 'input').glob(pattern))

        fastq_fns = [get_fastq_fns(which) for which in fastq.quartet_order]
        bam_fns = sorted((sample_dir / 'guide_mapping').glob('*.bam'))

        fn_quartets = (fastq.read_quartets(fns) for fns in zip(*fastq_fns))
        quartets = chain.from_iterable(fn_quartets)

        alignment_files = (pysam.AlignmentFile(str(fn)) for fn in bam_fns)
        mappings = chain.from_iterable(alignment_files)
        primary_mappings = (al for al in mappings if not al.is_secondary)

        sorters = {}
        guide_counts = Counter()

        for al, quartet in progress(zip(primary_mappings, quartets), total=stats['num_reads']):
            if al.query_name != quartet.I1.name.split(' ')[0]:
                raise ValueError
                
            if al.is_unmapped:
                guide = 'unknown'
            else:
                guide = al.reference_name

            if guide not in sorters:
                sorters[guide] = collapse.UMISorter(guide_dir / guide)

            sorters[guide].write(quartet)

            guide_counts[guide] += 1

        for sorter in progress(sorters.values()):
            sorter.close()

        pd.Series(guide_counts).to_csv(sample_dir / 'guide_counts.txt', sep='\t')
