import gzip
from pathlib import Path
from collections import Counter

import pandas as pd
import tqdm

import hits.fastq
import hits.utilities

progress = tqdm.tqdm

base_dir = Path('/lab/solexa_weissman/jah/projects/ddr' )
batch = '2020_10_13_miseq'

data_dir = base_dir  / 'data' / batch

sample_sheet_fn = data_dir / 'sample_sheet.csv'
sample_sheet = pd.read_csv(sample_sheet_fn, index_col='sample_name')

fns = {which: data_dir / f'{which}.fastq.gz' for which in ['R1', 'i7', 'i5']}
reads = {k: hits.fastq.reads(v) for k, v in fns.items()} 

counts = Counter()

resolvers = {k: hits.utilities.get_one_mismatch_resolver(sample_sheet[k]).get for k in ['i7', 'i5']}

sample_to_fh = {s: gzip.open(data_dir / fn, 'wt', compresslevel=1) for s, fn in sample_sheet['fastq_fn'].items()}

for R1, i7, i5 in progress(zip(reads['R1'], reads['i7'], reads['i5'])):
    i7_samples = resolvers['i7'](i7.seq, {'unknown'})
    i5_samples = resolvers['i5'](i5.seq, {'unknown'})
    
    consistent_with_both = i7_samples & i5_samples

    if len(consistent_with_both) == 1:
        sample = next(iter(consistent_with_both))
        if sample == 'unknown':
            sample = (i7.seq, i5.seq)

    elif len(consistent_with_both) > 1:
        print(i7.seq, i5.seq, consistent_with_both)
        raise ValueError
        
    counts[sample] += 1
                
    if sample in sample_to_fh:
        sample_to_fh[sample].write(str(R1))
        
for fh in sample_to_fh.values():
    fh.close()      

index_counts_fn = data_dir / 'index_counts.txt'
pd.Series(counts).sort_values(ascending=False).to_csv(index_counts_fn, header=None)