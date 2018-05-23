#!/usr/bin/env python3

import argparse
import array
import heapq
import subprocess
from collections import namedtuple, Counter
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import bokeh.palettes
import yaml
import tqdm
import pysam

progress = {
    True: tqdm.tqdm_notebook,
    False: tqdm.tqdm,
}

from sequencing import fastq, utilities, sw, sam
from sequencing import annotation as annotation_module

from collapse_cython import hq_mismatches_from_seed, hq_mismatches, hamming_distance

CELL_BC_TAG = 'CB'
UMI_TAG = 'UR'
NUM_READS_TAG = 'ZR'
CLUSTER_ID_TAG = 'ZC'

LOW_Q = 10
HIGH_Q = 31
N_Q = 2


cluster_fields = [
    ('cell_BC', 's'),
    ('UMI', 's'),
    ('num_reads', '06d'),
    ('cluster_id', 's'),
]
cluster_Annotation = annotation_module.Annotation_factory(cluster_fields)

def call_consensus(als, max_read_length=291):
    statistics = fastq.quality_and_complexity(als, max_read_length, alignments=True, min_q=30)
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

    majority = (best_stats['c'] / len(als)) > 0.5
    at_least_one_hq = best_stats['c_above_min_q'] > 0
    
    qs = np.full(max_read_length, LOW_Q, dtype=int)
    qs[majority & at_least_one_hq] = HIGH_Q
    
    ties = (best_stats == stat_tuples[rl_range, second_best_idxs])

    best_idxs[ties] = utilities.base_to_index['N']
    qs[ties] = N_Q

    consensus = pysam.AlignedSegment()
    consensus.query_sequence = ''.join(utilities.base_order[i] for i in best_idxs)
    consensus.query_qualities = array.array('B', qs)
    consensus.set_tag(NUM_READS_TAG, len(als), 'i')

    return consensus

def within_radius_of_seed(seed, als):
    seed_b = seed.encode()
    ds = [hq_mismatches_from_seed(seed_b, al.query_sequence.encode(), al.query_qualities, 20)
          for al in als]
    
    near_seed = []
    remaining = []
    
    for i, (d, al) in enumerate(zip(ds, als)):
        if d < 10:
            near_seed.append(al)
        else:
            remaining.append(al)
    
    return near_seed, remaining

def propose_seed(als):
    seq, count = Counter(al.query_sequence for al in als).most_common(1)[0]
    
    if count > 1:
        seed = seq
    else:
        seed = call_consensus(als).query_sequence
        
    return seed

def make_singleton_cluster(al):
    singleton = pysam.AlignedSegment()
    singleton.query_sequence = al.query_sequence
    singleton.query_qualities = al.query_qualities
    singleton.set_tag(NUM_READS_TAG, 1, 'i')
    return singleton

def form_clusters(als):
    if len(als) == 0:
        clusters = []
    
    elif len(als) == 1:
        clusters = [make_singleton_cluster(al) for al in als]
    
    else:
        seed = propose_seed(als)
        near_seed, remaining = within_radius_of_seed(seed, als)
        
        if len(near_seed) == 0:
            # didn't make progress, so give up
            clusters = [make_singleton_cluster(al) for al in als]
        
        else:
            clusters = [call_consensus(near_seed)] + form_clusters(remaining)
            
    return clusters

def hamming_distance_matrix(seqs):
    seq_bs = [seq.encode() for seq in seqs]
    
    n = len(seqs)
    ds = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            ds[i, j] = hamming_distance(seq_bs[i], seq_bs[j])

    return ds

def hq_hamming_distance_matrix(reads):
    seqs = [r.seq.encode() for r in reads]
    quals = [r.qual.encode() for r in reads]
    
    n = len(reads)
    ds = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            ds[i, j] = hq_mismatches(seqs[i], seqs[j], quals[i], quals[j], 20)

    return ds

def hq_levenshtein_distance_matrix(reads):
    n = len(reads)
    ds = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            indels, num_hq_mismatches = align_clusters(reads[i], reads[j])
            ds[i, j] = indels + num_hq_mismatches

    return ds

def align_clusters(first, second):
    al = sw.global_alignment(first.query_sequence, second.query_sequence)
    
    num_hq_mismatches = 0
    for q_i, t_i in al['mismatches']:
        if (first.query_qualities[q_i] > 20) and (second.query_qualities[t_i] > 20):
            num_hq_mismatches += 1
            
    return al['XO'], num_hq_mismatches

sort_key = lambda al: (al.get_tag(CELL_BC_TAG), al.get_tag(UMI_TAG))
empty_header = pysam.AlignmentHeader()

def sort_cellranger_bam(bam_fn, sorted_fn, notebook=True):
    bam_fh = pysam.AlignmentFile(str(bam_fn))
    total_reads = bam_fh.mapped + bam_fh.unmapped

    als = bam_fh

    relevant = (al for al in als if al.is_unmapped and al.has_tag(CELL_BC_TAG))
    
    chunk_fns = []
        
    for i, chunk in enumerate(utilities.chunks(relevant, 10000000)):
        suffix = '.{:06d}.bam'.format(i)
        chunk_fn = Path(sorted_fn).with_suffix(suffix)
        sorted_chunk = sorted(chunk, key=sort_key)
    
        with pysam.AlignmentFile(str(chunk_fn), 'wb', header=empty_header) as fh:
            for al in sorted_chunk:
                fh.write(al)

        chunk_fns.append(chunk_fn)

    chunk_fhs = [pysam.AlignmentFile(str(fn), check_sq=False) for fn in chunk_fns]
    
    with pysam.AlignmentFile(str(sorted_fn), 'wb', header=empty_header) as fh:
        for al in heapq.merge(*chunk_fhs, key=sort_key):
            fh.write(al)

    for fn in chunk_fns:
        fn.unlink()
    
    pysam.index(str(sorted_fn))

def error_correct_UMIs(cell_group):
    UMI_counts = Counter(s.UMI for s in cell_group)
    UMIs = [UMI for UMI, count in UMI_counts.most_common()]

    ds = hamming_distance_matrix(UMIs)

    corrections = {}

    # Moving from least common to most common, register a correction
    # from a UMI to the most common UMI that is within Hamming distance
    # 1 of it.
    for j in range(len(ds) - 1, -1, -1):
        for i in range(j - 1, -1, -1):
            if ds[i, j] == 1:
                corrections[UMIs[j]] = UMIs[i]

    # If a correction points to a UMI that is itself going to be corrected,
    # propogate this correction through.  
    for from_, to in list(corrections.items()):
        while to in corrections:
            to = corrections[to]

        corrections[from_] = to
    
    for singleton in cell_group:
        if singleton.UMI in corrections:
            singleton.UMI = corrections[singleton.UMI]
    
    return cell_group

def merge_annotated_clusters(biggest, other):
    merged_id = biggest.get_tag(CLUSTER_ID_TAG)
    if not merged_id.endswith('+'):
        merged_id = merged_id + '+'
    biggest.set_tag(CLUSTER_ID_TAG, merged_id, 'Z')

    total_reads = biggest.get_tag(NUM_READS_TAG) + other.get_tag(NUM_READS_TAG)
    biggest.set_tag(NUM_READS_TAG, total_reads, 'i')

    return biggest

def form_collapsed_clusters(sorted_fn, collapsed_fn, notebook=True):
    sorted_als = pysam.AlignmentFile(str(sorted_fn), check_sq=False)
    total_reads = sorted_als.unmapped
    
    #sorted_als = progress[False](sorted_als, desc='Collapsing', total=total_reads)
    #sorted_als = itertools.islice(sorted_als, 1000000)

    groups = utilities.group_by(sorted_als, sort_key)
    
    with pysam.AlignmentFile(str(collapsed_fn), 'wb', header=empty_header) as collapsed_fh:
        for (cell_BC, umi), group in groups:
            annotated_clusters = []

            clusters = form_clusters(group)
            clusters = sorted(clusters, key=lambda c: c.get_tag(NUM_READS_TAG), reverse=True)

            for i, cluster in enumerate(clusters):
                cluster.set_tag(CELL_BC_TAG, cell_BC, 'Z')
                cluster.set_tag(UMI_TAG, umi, 'Z')
                cluster.set_tag(CLUSTER_ID_TAG, str(i), 'Z')

            biggest = clusters[0]
            rest = clusters[1:]

            not_collapsed = []

            for other in rest:
                if other.get_tag(NUM_READS_TAG) == biggest.get_tag(NUM_READS_TAG):
                    not_collapsed.append(other)
                else:
                    indels, hq_mismatches = align_clusters(biggest, other)

                    if indels <= 2 and hq_mismatches <= 5:
                        biggest = merge_annotated_clusters(biggest, other)
                    else:
                        not_collapsed.append(other)
            
            for cluster in [biggest] + not_collapsed:
                annotation = cluster_Annotation(cell_BC=cluster.get_tag(CELL_BC_TAG),
                                                UMI=cluster.get_tag(UMI_TAG),
                                                num_reads=cluster.get_tag(NUM_READS_TAG),
                                                cluster_id=cluster.get_tag(CLUSTER_ID_TAG),
                                               )

                cluster.query_name = str(annotation)
                collapsed_fh.write(cluster)

    pysam.index(str(collapsed_fn))

def split_into_guide_fastqs(collapsed_fn, cell_BC_to_guide, gemgroup, group_dir):
    clusters = pysam.AlignmentFile(str(collapsed_fn), check_sq=False)

    guide_fhs = {}

    for cluster in clusters:
        cell_BC = cluster.get_tag(CELL_BC_TAG)
        cell_BC = '{0}-{1}'.format(cell_BC.split('-')[0], gemgroup)
        guide = cell_BC_to_guide.get(cell_BC, 'unknown')
        if guide == '*':
            guide = 'unknown'

        if guide not in guide_fhs:
            guide_fn = (Path(group_dir) / guide).with_suffix('.fastq')
            guide_fhs[guide] = guide_fn.open('w')

        read = sam.mapping_to_Read(cluster)

        # temporary hack
        read.name = '{0}_{1}'.format(cell_BC, read.name.split('_', 1)[1])

        guide_fhs[guide].write(str(read))

    for guide, fh in guide_fhs.items():
        fh.close()

    guides = sorted(guide_fhs)
    return guides

def make_sample_sheet(group_dir, target, guides):
    color_list = bokeh.palettes.Category20c_20[:16] #+ bokeh.palettes.Category20b_20
    color_groups = itertools.cycle(list(zip(*[iter(color_list)]*4)))

    sample_sheet = {}

    grouped_guides = utilities.group_by(sorted(guides), lambda n: n.split('-')[0])
    for (group_name, group), color_group in zip(grouped_guides, color_groups):
        for name, color in zip(group, color_group[1:]):
            sample_sheet[name] = {
                'fastq_fns': name + '.fastq',
                'target_info': target,
                'project': 'screen',
                'color': color,
            }

    sample_sheet_fn = group_dir / 'sample_sheet.yaml'
    sample_sheet_fn.write_text(yaml.dump(sample_sheet, default_flow_style=False))

def make_cluster_fastqs(collapsed_fn, target, gemgroup, notebook=True):
    group_dir = Path(collapsed_fn).parent
    df = pd.read_csv('/home/jah/projects/britt/data/cell_identities.csv', index_col='cell_barcode') 
    cell_BC_to_guide = df['guide_identity']
    guides = split_into_guide_fastqs(collapsed_fn, cell_BC_to_guide, gemgroup, group_dir)
    make_sample_sheet(group_dir, target, guides)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--cellranger_dir', required=True)
    mode_group = parser.add_mutually_exclusive_group(required=True)

    mode_group.add_argument('--collapse', nargs=4, metavar=('INPUT_FN', 'OUTPUT_NAME', 'GEMGROUP', 'TARGET'))
    mode_group.add_argument('--parallel', metavar='MAX_PROCS')

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    cellranger_dir = Path(args.cellranger_dir)

    if args.parallel is not None:
        sample_sheet_fn = base_dir / 'data' / 'sample_sheet.yaml'
        sample_sheet = yaml.load(sample_sheet_fn.read_text())

        arg_tuples = []
        for name, info in sorted(sample_sheet.items()):
            input_fn = cellranger_dir / name / 'outs' / 'possorted_genome_bam.bam'
            arg_tuples.append((str(input_fn), info['name'], str(info['gemgroup']), info['target']))

        parallel_command = [
            'parallel',
            '-n', '4', 
            '--verbose',
            '--max-procs', args.parallel,
            './collapse.py',
            '--collapse', ':::',
        ]

        for arg_tuple in arg_tuples:
            parallel_command.extend(arg_tuple)
    
        subprocess.check_call(parallel_command)

    elif args.collapse is not None:
        input_fn, output_name, gemgroup, target = args.collapse

        sorted_fn = (base_dir / 'data' / output_name / output_name).with_suffix('.bam')
        if not Path(sorted_fn).exists():
            sort_cellranger_bam(input_fn, sorted_fn)

        collapsed_fn = sorted_fn.with_name(sorted_fn.stem + '_collapsed.bam')
        if not Path(collapsed_fn).exists():
            form_collapsed_clusters(sorted_fn, collapsed_fn)
        form_collapsed_clusters(sorted_fn, collapsed_fn)
            
        #make_cluster_fastqs(collapsed_fn, target, gemgroup)
