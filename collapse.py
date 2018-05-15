import argparse
from collections import namedtuple, Counter
from itertools import cycle
from pathlib import Path

import numpy as np
import bokeh.palettes
import yaml
import tqdm

progress = {
    True: tqdm.tqdm_notebook,
    False: tqdm.tqdm,
}

from sequencing import fastq, utilities, sw
from sequencing import annotation as annotation_module

from collapse_cython import hq_mismatches_from_seed, hq_mismatches, hamming_distance

low_q = fastq.encode_sanger([10])
high_q = fastq.encode_sanger([31])
N_q = fastq.encode_sanger([2])

def call_consensus(reads):
    read_length = len(reads[0].seq)
    statistics = fastq.quality_and_complexity(reads, read_length, min_q=30)
    shape = statistics['c'].shape

    rl_range = np.arange(read_length)
    
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
    
    qs = np.full(read_length, low_q, dtype='S1')
    qs[majority & at_least_one_hq] = high_q
    
    ties = (best_stats == stat_tuples[rl_range, second_best_idxs])
    best_idxs[ties] = utilities.base_to_index['N']
    qs[ties] = N_q

    seq = ''.join(utilities.base_order[i] for i in best_idxs)
    qual = b''.join(qs).decode()

    consensus_read = fastq.Read('', seq, qual)
    consensus = AnnotatedRead(consensus_read, {'num_reads': len(reads)})

    return consensus

def within_radius_of_seed(seed, reads):
    seed_b = seed.encode()
    
    seqs = [r.seq.encode() for r in reads]
    quals = [r.qual.encode() for r in reads]
    
    ds = [hq_mismatches_from_seed(seed_b, seq, qual, 20) for seq, qual in zip(seqs, quals)]
    
    near_seed = []
    remaining = []
    
    for i, (d, read) in enumerate(zip(ds, reads)):
        if d < 10:
            near_seed.append(read)
        else:
            remaining.append(read)
    
    return near_seed, remaining

def propose_seed(reads):
    seq, count = Counter(r.seq for r in reads).most_common(1)[0]
    
    if count > 1:
        seed = seq
    else:
        seed = call_consensus(reads).seq
        
    return seed

def form_clusters(reads):
    if len(reads) == 0:
        clusters = []
    
    elif len(reads) == 1:
        clusters = [AnnotatedRead(r, {'num_reads': 1}) for r in reads]
    
    else:
        seed = propose_seed(reads)
        near_seed, remaining = within_radius_of_seed(seed, reads)
        
        if len(near_seed) == 0:
            # didn't make progress, so give up
            clusters = [AnnotatedRead(r, {'num_reads': 1}) for r in reads]
        
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

q20 = fastq.encode_sanger([20])

def align_clusters(first, second):
    al = sw.global_alignment(first.seq, second.seq)
    
    num_hq_mismatches = 0
    for q_i, t_i in al['mismatches']:
        if (first.qual[q_i] > q20) and (second.qual[t_i] > q20):
            num_hq_mismatches += 1
            
    return al['XO'], num_hq_mismatches

input_fields = [
    ('original_name', 's'),
    ('cell_BC', 's'),
    ('UMI', 's'),
    ('eqc_count', 'd'),
    ('guide', 's'),
    ('confidence', 's'),
    ('guides_in_cell', 'd'),
]

input_Annotation = annotation_module.Annotation_factory(input_fields, read_only=True)

cluster_fields = [
    ('cell_BC', 's'),
    ('UMI', 's'),
    ('cluster_id', 's'),
    ('num_reads', '05d'),
    ('guide', 's'),
]

cluster_Annotation = annotation_module.Annotation_factory(cluster_fields)

class AnnotatedRead(object):
    def __init__(self, read, annotation):
        self.name = read.name
        self.seq = read.seq
        self.qual = read.qual
        self.annotation = annotation

    @classmethod
    def from_Annotation(cls, read, Annotation):
        annotation = Annotation.from_identifier(read.name)
        return cls(read, annotation)

    def to_record(self, Annotation):
        name = Annotation.from_annotation(self.annotation)
        return str(fastq.Read(name, self.seq, self.qual))

    def __getattr__(self, name):
        return self.annotation[name]

def make_sorted_singletons(input_fastq_fns, notebook=True):
    #import itertools
    reads = fastq.reads(input_fastq_fns)
    #reads = itertools.islice(reads, 10000)
    
    singletons = []

    for read in progress[notebook](reads, desc='Reading input'):
        singleton = AnnotatedRead.from_Annotation(read, input_Annotation)
        for i in range(singleton.eqc_count):
            singletons.append(singleton)
   
    singletons = sorted(singletons, key=lambda s: (s.cell_BC, s.UMI))

    return singletons

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
    merged_id = '{0}+{1}'.format(biggest.cluster_id, other.cluster_id)
    total_reads = biggest.num_reads + other.num_reads
    biggest.annotation['cluster_id'] = merged_id
    biggest.annotation['num_reads'] = total_reads
    return biggest

def form_collapsed_clusters(sorted_singletons, notebook=True):
    sorted_singletons = progress[notebook](sorted_singletons, desc='Collapsing')
    sort_key = lambda s: (s.cell_BC, s.guide, s.UMI)
    groups = utilities.group_by(sorted_singletons, sort_key)
    
    collapsed_clusters = []
    
    for (cell_BC, guide, UMI), group in groups:
        annotated_clusters = []

        clusters = form_clusters(group)
        clusters = sorted(clusters, key=lambda c: c.num_reads, reverse=True)

        annotated_clusters = []

        for i, cluster in enumerate(clusters):
            annotation = {
                'cell_BC': cell_BC,
                'UMI': UMI,
                'guide': guide,
                'cluster_id': str(i),
                'num_reads': cluster.num_reads,
            }
            annotated_clusters.append(AnnotatedRead(cluster, annotation))

        biggest = annotated_clusters[0]
        rest = annotated_clusters[1:]

        if len(annotated_clusters) == 1:
            collapsed_clusters.append(biggest)

        else:
            not_collapsed = []

            for other in rest:
                if other.num_reads == biggest.num_reads:
                    not_collapsed.append(other)
                else:
                    indels, hq_mismatches = align_clusters(biggest, other)

                    if indels <= 2 and hq_mismatches <= 5:
                        biggest = merge_annotated_clusters(biggest, other)
                    else:
                        not_collapsed.append(other)
            
            collapsed_clusters.append(biggest)
            collapsed_clusters.extend(not_collapsed)

    return collapsed_clusters

def split_into_guide_fastqs(base_dir, clusters, notebook=True):
    guide_fhs = {}

    for cluster in progress[notebook](clusters, desc='Writing'):
        guide = cluster.guide

        if guide not in ('EMPTY', '*'):
            if guide not in guide_fhs:
                guide_fn = (base_dir / guide).with_suffix('.fastq')
                guide_fhs[guide] = open(str(guide_fn), 'w')

            record = cluster.to_record(cluster_Annotation)
            guide_fhs[guide].write(record)

    for guide, fh in guide_fhs.items():
        fh.close()

    guides = sorted(guide_fhs)
    return guides

def make_sample_sheet(base_dir, target, guides):
    color_list = bokeh.palettes.Category20c_20[:16] #+ bokeh.palettes.Category20b_20
    color_groups = cycle(list(zip(*[iter(color_list)]*4)))

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

    sample_sheet_fn = base_dir / 'sample_sheet.yaml'
    sample_sheet_fn.write_text(yaml.dump(sample_sheet, default_flow_style=False))

def make_cluster_fastqs(base_dir, target, notebook=True):
    base_dir = Path(base_dir)
    input_fastq_fns = list(base_dir.glob('*.gz'))

    sorted_singletons = make_sorted_singletons(input_fastq_fns, notebook)
    #sorted_singletons = error_correct_UMIs(sorted_singletons, notebook)

    clusters = form_collapsed_clusters(sorted_singletons, notebook)

    guides = split_into_guide_fastqs(base_dir, clusters, notebook)
    make_sample_sheet(base_dir, target, guides)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--target', required=True)

    args = parser.parse_args()

    make_cluster_fastqs(Path(args.base_dir), args.target, notebook=False)
