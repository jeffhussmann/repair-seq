from collections import Counter
from pathlib import Path

import pandas as pd
import tqdm
import yaml

import hits.fastq
import knock_knock.target_info
import repair_seq.guide_library

def count_guides(base_dir, batch, sample_name):
    ''' target_info should have a feature annotating where the sequencing primer anneals,
    and this feature should be followed immediately by protospacer sequence.
    '''

    data_dir = Path(base_dir) / 'data' / batch
    sample_sheet_fn = data_dir / 'sample_sheet.yaml'
    sample_sheet = yaml.safe_load(sample_sheet_fn.read_text())

    sample_info = sample_sheet['samples'][sample_name]

    ti = knock_knock.target_info.TargetInfo(base_dir, sample_info['target_info'],
                                            sequencing_start_feature_name=sample_info['guide_primer'],
                                           )

    guide_library = repair_seq.guide_library.GuideLibrary(base_dir, sample_info['guide_library'])

    guide_primer_seq = ti.feature_sequence(ti.target, sample_info['guide_primer'])
    guide_primer_length = len(guide_primer_seq)

    protospacer_lengths = guide_library.guides_df['protospacer'].str.len()
    max_protospacer_length = max(protospacer_lengths)
    length_to_examine = guide_primer_length + max_protospacer_length

    guide_seq_resolver = {s[:length_to_examine]: g for g, s in guide_library.full_guide_seqs.items()}.get

    primer_prefix_length = 6
    prefix = guide_primer_seq[:primer_prefix_length]

    guide_counts = Counter()

    fastq_fn = data_dir / sample_info['guide_fastq_fn']

    total = sample_info.get('num_reads')
    for read in tqdm.tqdm(hits.fastq.reads(fastq_fn), total=total):
        try:
            start = read.seq.index(prefix, 0, primer_prefix_length + 4)
        except:
            start = 0

        trimmed = read[start:]

        guide_seq = trimmed.seq[:length_to_examine]

        guide = guide_seq_resolver(guide_seq, 'unknown')

        guide_counts[guide] += 1

    guide_counts = pd.Series(guide_counts).reindex(guide_library.guides).fillna(0).astype(int)

    guide_counts.name = 'read_count'
    guide_counts.index.name = 'guide'
    
    csv_fn = data_dir / f'{sample_name}_guide_counts.csv'
    guide_counts.to_csv(csv_fn)

    return guide_counts

def load_guide_counts(base_dir, batch, sample_name):
    data_dir = Path(base_dir) / 'data' / batch
    csv_fn = data_dir / f'{sample_name}_guide_counts.csv'
    guide_counts = pd.read_csv(csv_fn, index_col=0).squeeze()
    return guide_counts

def load_batch_guide_counts(base_dir, batch):
    data_dir = Path(base_dir) / 'data' / batch
    sample_sheet_fn = data_dir / 'sample_sheet.yaml'
    sample_sheet = yaml.safe_load(sample_sheet_fn.read_text())

    all_counts = {}
    for sample_name in sample_sheet['samples']:
        all_counts[sample_name] = load_guide_counts(base_dir, batch, sample_name) 

    return pd.DataFrame(all_counts)
