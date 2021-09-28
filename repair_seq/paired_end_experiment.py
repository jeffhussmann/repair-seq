import gzip
from collections import Counter

import pandas as pd

from hits import utilities, fastq

memoized_property = utilities.memoized_property

import knock_knock.illumina_experiment
import repair_seq.pooled_layout

class PairedEndExperiment(knock_knock.illumina_experiment.IlluminaExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.outcome_fn_keys = [
            'outcome_list',
            #'no_overlap_outcome_list',
        ]

        self.diagram_kwargs.update(draw_sequence=True,
                                   flip_target=self.target_info.sequencing_direction == '-',
                                   highlight_SNPs=True,
                                   center_on_primers=True,
                                   split_at_indels=True,
                                  )

        self.read_types = [
            'stitched',
            'stitched_by_name',
            'nonredundant',
            'R1_no_overlap',
            'R2_no_overlap',
        ]

        self.error_corrected = False
        self.layout_mode = 'amplicon'

    @property
    def preprocessed_read_type(self):
        return 'stitched_by_name'

    @property
    def default_read_type(self):
        return 'stitched_by_name'

    @property
    def read_types_to_align(self):
        return [
            'nonredundant',
            #'R1_no_overlap',
            #'R2_no_overlap',
            ]

    @memoized_property
    def categorizer(self):
        #return repair_seq.pooled_layout.Layout
        return repair_seq.prime_editing_layout.Layout

    @memoized_property
    def max_relevant_length(self):
        return 1500

    @property
    def read_pairs(self):
        read_pairs = fastq.read_pairs(self.fns['R1'], self.fns['R2'], standardize_names=True)
        return read_pairs

    def sort_stitched_read_pairs(self):
        reads = sorted(self.reads_by_type('stitched'), key=lambda read: read.name)
        fn = self.fns_by_read_type['fastq']['stitched_by_name']
        with gzip.open(fn, 'wt', compresslevel=1) as sorted_fh:
            for read in reads:
                sorted_fh.write(str(read))

    def preprocess(self):
        self.stitch_read_pairs()
        self.sort_stitched_read_pairs()

    def process(self, stage):
        try:
            if stage == 'preprocess':
                self.preprocess()

            elif stage == 'align':
                self.make_nonredundant_sequence_fastq()
                
                for read_type in self.read_types_to_align:
                    self.generate_alignments(read_type)
                    self.generate_supplemental_alignments_with_STAR(read_type, min_length=20)
                    self.combine_alignments(read_type)

            elif stage == 'categorize':
                self.categorize_outcomes()

                if 'R1_no_overlap' in self.read_types_to_align:
                    self.categorize_no_overlap_outcomes()

                self.count_read_lengths()
                self.count_outcomes()

                self.record_sanitized_category_names()
            
            elif stage == 'visualize':
                self.generate_figures()
        except:
            print(self.group, self.sample_name)
            raise
