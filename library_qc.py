#!/usr/bin/env python3.6

import argparse
import multiprocessing
from collections import defaultdict, Counter
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
import pysam

from hits import mapping_tools, sam
from hits.utilities import memoized_property

class Sample():
    def __init__(self, base_dir, group, name):
        self.base_dir = Path(base_dir)
        self.group = group
        self.name = name

        data_dir = self.base_dir / 'data' / group

        yaml_fn = data_dir / 'sample_sheet.yaml'
        sample_sheet = yaml.safe_load(yaml_fn.read_text())

        results_dir = self.base_dir / 'results' / group / name
        results_dir.mkdir(exist_ok=True, parents=True)

        R1_fn = data_dir / sample_sheet[name]

        self.fns = {
            'R1': R1_fn,
            'guides': '/home/jah/projects/ddr/guides/DDR_sublibrary/guides.txt',
            'guides_STAR_index': '/home/jah/projects/ddr/guides/DDR_sublibrary/STAR_index',

            'STAR_output_prefix': results_dir / 'alignments.',
            'bam': results_dir / 'alignments.bam',
            'bad_guides': results_dir / 'bad_guides.bam',
            'bad_guides_by_name': results_dir / 'bad_guides.by_name.bam',

            'indel_distributions': results_dir / 'indel_distributions.txt',
            'edit_distance_distributions': results_dir / 'edit_distance_distributions.txt',
            'guide_counts': results_dir / 'guide_counts.txt',
        }

    def align_reads(self):
        mapping_tools.map_STAR(self.fns['R1'],
                               self.fns['guides_STAR_index'],
                               self.fns['STAR_output_prefix'],
                               sort=False,
                               mode='guide_alignment',
                               include_unmapped=True,
                               bam_fn=self.fns['bam'],
                              )

    def count_alignments(self):
        header = sam.get_header(self.fns['bam'])

        mappings = pysam.AlignmentFile(self.fns['bam'])
        mapping_groups = sam.grouped_by_name(mappings)

        read_length = 50

        edit_distance = defaultdict(lambda: np.zeros(read_length + 1, int))
        indels = defaultdict(lambda: np.zeros(read_length + 1, int))

        def edit_info(al):
            if al.is_unmapped or al.is_reverse:
                return (read_length, read_length)
            else:
                return (sam.total_indel_lengths(al), al.get_tag('NM'))

        guide_counts = Counter()

        bad_guides_by_pos = sam.AlignmentSorter(self.fns['bad_guides'], header)
        bad_guides_by_name = sam.AlignmentSorter(self.fns['bad_guides_by_name'], header, by_name=True)

        with bad_guides_by_pos, bad_guides_by_name:
            for query_name, als in mapping_groups:
                edit_tuples = [(al, edit_info(al)) for al in als]

                for al, (num_indels, NM) in edit_tuples:
                    guide = al.reference_name
                    edit_distance[guide][NM] += 1
                    indels[guide][num_indels] += 1

                min_edit = min(info for al, info in edit_tuples)
                min_edit_als = [al for al, info in edit_tuples if info == min_edit]
                    
                min_indels, min_NM = min_edit
                if min_indels == 0 and min_NM <= 1 and len(min_edit_als) == 1:
                    al = min_edit_als[0]
                    guide = al.reference_name
                else:
                    guide = 'unknown'
                    for al in als:
                        bad_guides_by_pos.write(al)
                        bad_guides_by_name.write(al)
            
                guide_counts[guide] += 1
        
        pd.DataFrame(indels).T.to_csv(self.fns['indel_distributions'])
        pd.DataFrame(edit_distance).T.to_csv(self.fns['edit_distance_distributions'])

        guide_counts = pd.Series(guide_counts).sort_index()
        guide_counts.to_csv(self.fns['guide_counts'])

    @memoized_property
    def guide_counts(self):
        counts = pd.read_csv(self.fns['guide_counts'], squeeze=True, index_col=0, header=None)
        return counts

    @memoized_property
    def guides(self):
        return pd.read_csv(self.fns['guides'], sep='\t', index_col='short_name')

class SampleGroup():
    def __init__(self, base_dir, group):
        data_dir = Path(base_dir) / 'data' / group
        yaml_fn = data_dir / 'sample_sheet.yaml'

        sample_sheet = yaml.safe_load(yaml_fn.read_text())

        self.samples = [Sample(base_dir, group, name) for name in sample_sheet]

    @memoized_property
    def guide_counts(self):
        return pd.DataFrame({s.name: s.guide_counts for s in self.samples})

def process_sample(sample):
    sample.align_reads()
    sample.count_alignments()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('base_dir', type=Path)
    parser.add_argument('group', type=str)
    parser.add_argument('max_procs', type=int)
        
    args = parser.parse_args()

    sample_group = SampleGroup(args.base_dir, args.group)

    pool = multiprocessing.Pool(processes=args.max_procs)
    pool.map(process_sample, sample_group.samples)
