import array
from collections import Counter

import numpy as np
import tqdm
import pysam

from hits import fastq, utilities
from hits import annotation as annotation_module

from .collapse_cython import hq_mismatches_from_seed

progress = tqdm.tqdm_notebook

NUM_READS_TAG = 'ZR'
CLUSTER_ID_TAG = 'ZC'

HIGH_Q = 31
LOW_Q = 10
N_Q = 2

annotation_fields = {
    'UMI': [
        ('UMI', 's'),
        ('original_name', 's'),
    ],

    'UMI_guide': [
        ('UMI', 's'),
        ('guide', 's'),
        ('guide_qual', 's'),
        ('original_name', 's'),
    ],

    'collapsed_UMI': [
        ('UMI', 's'),
        ('guide', 's'),
        ('guide_qual', 's'),
        ('cluster_id', '06d'),
        ('num_reads', '06d'),
    ],

    'collapsed_UMI_mismatch': [
        ('UMI', 's'),
        ('cluster_id', '06d'),
        ('num_reads', '010d'),
        ('mismatch', 'd'),
    ],

    'common_sequence': [
        ('rank', '012d'),
        ('count', '012d'),
    ],

    'R2_with_guide': [
        ('query_name', 's'),
        ('guide', 's'),
        ('guide_qual', 's'),
    ],

    'R2_with_guide_mismatches': [
        ('query_name', 's'),
        ('mismatches', 's'),
    ],
}

Annotations = {key: annotation_module.Annotation_factory(fields) for key, fields in annotation_fields.items()}

def consensus_seq_and_qs(reads, max_read_length, bam):
    if max_read_length is None:
        max_read_length = len(reads[0].query_sequence)

    statistics = fastq.quality_and_complexity(reads, max_read_length, alignments=bam, min_q=30)
    shape = statistics['c'].shape

    rl_range = np.arange(max_read_length)
    
    fields = [
        ('c_above_min_q', int),
        ('c', int),
        ('average_q', float),
    ]

    stat_tuples = np.zeros(shape, dtype=fields)
    for k in ['c_above_min_q', 'c', 'average_q']:
        stat_tuples[k] = statistics[k]

    argsorted = stat_tuples.argsort()
    second_best_idxs, best_idxs = argsorted[:, -2:].T
    
    best_stats = stat_tuples[rl_range, best_idxs]

    majority = (best_stats['c'] / len(reads)) > 0.5
    at_least_one_hq = best_stats['c_above_min_q'] > 0
    
    qs = np.full(max_read_length, LOW_Q, dtype=int)
    qs[majority & at_least_one_hq] = HIGH_Q
    
    ties = (best_stats == stat_tuples[rl_range, second_best_idxs])

    best_idxs[ties] = utilities.base_to_index['N']
    qs[ties] = N_Q

    seq = ''.join(utilities.base_order[i] for i in best_idxs)

    return seq, qs

def call_consensus(reads, max_read_length, bam):
    seq, qs = consensus_seq_and_qs(reads, max_read_length, bam)

    if bam:
        consensus = pysam.AlignedSegment()
        consensus.query_sequence = seq
        consensus.query_qualities = array.array('B', qs)
        consensus.set_tag(NUM_READS_TAG, len(reads), 'i')
    else:
        guide_reads = []
        for read in reads:
            annotation = Annotations['UMI_guide'].from_identifier(read.name)
            guide_read = fastq.Read('PH', annotation['guide'], annotation['guide_qual'])
            guide_reads.append(guide_read)

        guide_seq, guide_qs = consensus_seq_and_qs(guide_reads, None, False)
        guide_qual = fastq.encode_sanger(guide_qs)

        annotation = Annotations['collapsed_UMI'](UMI='PH',
                                                  num_reads=len(reads),
                                                  guide=guide_seq,
                                                  guide_qual=guide_qual,
                                                  cluster_id=0,
                                                 )
        name = str(annotation)
        qual = fastq.encode_sanger(qs)
        consensus = fastq.Read(name, seq, qual)

    return consensus

def within_radius_of_seed(seed, reads, max_hq_mismatches):
    seed_b = seed.encode()
    ds = [hq_mismatches_from_seed(seed_b, read.query_sequence.encode(), read.query_qualities, 20)
          for read in reads]
    
    near_seed = []
    remaining = []
    
    for i, (d, al) in enumerate(zip(ds, reads)):
        if d <= max_hq_mismatches:
            near_seed.append(al)
        else:
            remaining.append(al)
    
    return near_seed, remaining

def propose_seed(reads, max_read_length, bam):
    seqs = (read.query_sequence for read in reads)

    seq_counts = Counter(seqs).most_common()

    highest_count = seq_counts[0][1]
    most_frequents = [s for s, c in seq_counts if c == highest_count]

    # If there is a tie, take the alphabetically first for determinism.
    seq = sorted(most_frequents)[0]
    
    if highest_count > 1:
        seed = seq
    else:
        consensus = call_consensus(reads, max_read_length, bam)
        seed = consensus.query_sequence
        
    return seed

def make_singleton_cluster(read, bam):
    if bam:
        singleton = pysam.AlignedSegment()
        singleton.query_sequence = read.query_sequence
        singleton.query_qualities = read.query_qualities
        singleton.set_tag(NUM_READS_TAG, 1, 'i')
    else:
        annotation = Annotations['UMI_guide'].from_identifier(read.name)
        name = Annotations['collapsed_UMI'](UMI=annotation['UMI'],
                                            guide=annotation['guide'],
                                            guide_qual=annotation['guide_qual'],
                                            cluster_id=0,
                                            num_reads=1,
                                           )
        singleton = fastq.Read(str(name), read.seq, read.qual)

    return singleton

def form_clusters(reads, max_read_length=None, max_hq_mismatches=0, bam=False):
    if len(reads) == 0:
        clusters = []
    
    elif len(reads) == 1:
        clusters = [make_singleton_cluster(read, bam) for read in reads]
    
    else:
        seed = propose_seed(reads, max_read_length, bam)
        near_seed, remaining = within_radius_of_seed(seed, reads, max_hq_mismatches)
        
        if len(near_seed) == 0:
            # didn't make progress, so give up
            clusters = [make_singleton_cluster(read, bam) for read in reads]
        
        else:
            consensus_near_seed = call_consensus(near_seed, max_read_length, bam)
            all_others = form_clusters(remaining, max_read_length, max_hq_mismatches, bam)
            clusters = [consensus_near_seed] + all_others
            
    return clusters