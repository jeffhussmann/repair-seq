import gzip
import itertools
from pathlib import Path
from collections import Counter
from contextlib import ExitStack

import pandas as pd
import tqdm

import hits.fastq
import hits.utilities

progress = tqdm.tqdm

def demux_SE(base_dir, batch, payload_read_type='R1', num_reads=None, only_first_n=None):
    base_dir = Path(base_dir)
    data_dir = base_dir  / 'data' / batch

    sample_sheet_fn = data_dir / 'sample_sheet.csv'
    sample_sheet = pd.read_csv(sample_sheet_fn, index_col='sample_name')

    fns = {which: data_dir / f'{which}.fastq.gz' for which in [payload_read_type, 'i7', 'i5']}
    reads = {k: hits.fastq.reads(v) for k, v in fns.items()} 

    counts = Counter()

    resolvers = {k: hits.utilities.get_one_mismatch_resolver(sample_sheet[k]).get for k in ['i7', 'i5']}

    zipped_reads = zip(reads[payload_read_type], reads['i7'], reads['i5'])

    if only_first_n is not None:
        zipped_reads = itertools.islice(zipped_reads, only_first_n)
        if num_reads is not None:
            num_reads = min(num_reads, only_first_n)
        else:
            num_reads = only_first_n

    with ExitStack() as stack:

        sample_to_fh = {}

        for sample, fn in sample_sheet['fastq_fn'].items():
            fh = stack.enter_context(gzip.open(data_dir / fn, 'wt', compresslevel=1))
            sample_to_fh[sample] = fh

        for payload_read, i7, i5 in progress(zipped_reads, total=num_reads):
            i7_samples = resolvers['i7'](i7.seq, {'unknown'})
            i5_samples = resolvers['i5'](i5.seq, {'unknown'})
            
            consistent_with_both = i7_samples & i5_samples

            if len(consistent_with_both) == 0:
                sample = f'{i7.seq}+{i5.seq}'

            elif len(consistent_with_both) == 1:
                sample = next(iter(consistent_with_both))
                if sample == 'unknown':
                    sample = f'{i7.seq}+{i5.seq}'

            else:
                print(i7.seq, i5.seq, consistent_with_both)
                raise ValueError
                
            counts[sample] += 1
                        
            if sample in sample_to_fh:
                sample_to_fh[sample].write(str(payload_read))
            
    index_counts_fn = data_dir / 'index_counts.txt'
    pd.Series(counts).sort_values(ascending=False).to_csv(index_counts_fn, header=None)

def demux_PE(base_dir, batch):
    data_dir = base_dir  / 'data' / batch

    sample_sheet_fn = data_dir / 'sample_sheet.csv'
    sample_sheet = pd.read_csv(sample_sheet_fn, index_col='sample_name')

    fns = {which: data_dir / f'{which}.fastq.gz' for which in ['R1', 'R2', 'i7', 'i5']}
    reads = {k: hits.fastq.reads(v) for k, v in fns.items()} 

    counts = Counter()

    resolvers = {k: hits.utilities.get_one_mismatch_resolver(sample_sheet[k]).get for k in ['i7', 'i5']}

    sample_to_fhs = {
        sample_name: {
            which: gzip.open(data_dir / sample_sheet.loc[sample_name, f'{which}_fn'], 'wt', compresslevel=1)
            for which in ['R1', 'R2']
        } for sample_name in sample_sheet.index
    }

    quartets = zip(reads['R1'], reads['R2'], reads['i7'], reads['i5'])
    quartets = itertools.islice(quartets, int(1e6))
    for R1, R2, i7, i5 in progress(quartets):
        i7_samples = resolvers['i7'](i7.seq, {'unknown'})
        i5_samples = resolvers['i5'](i5.seq, {'unknown'})
        
        consistent_with_both = i7_samples & i5_samples

        if len(consistent_with_both) == 0:
            sample = f'{i7.seq}+{i5.seq}'

        elif len(consistent_with_both) == 1:
            sample = next(iter(consistent_with_both))
            if sample == 'unknown':
                sample = f'{i7.seq}+{i5.seq}'

        else:
            print(i7.seq, i5.seq, consistent_with_both)
            raise ValueError
            
        counts[sample] += 1
                    
        if sample in sample_to_fhs:
            sample_to_fhs[sample]['R1'].write(str(R1))
            sample_to_fhs[sample]['R2'].write(str(R2))
            
    for fhs in sample_to_fhs.values():
        for fh in fhs.values():
            fh.close()      

    index_counts_fn = data_dir / 'index_counts.txt'
    pd.Series(counts).sort_values(ascending=False).to_csv(index_counts_fn, header=None)