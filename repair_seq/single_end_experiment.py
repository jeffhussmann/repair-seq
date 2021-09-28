import gzip
from collections import Counter

import pandas as pd

from hits import utilities, fastq, adapters

memoized_property = utilities.memoized_property

import knock_knock.experiment
import repair_seq.pooled_layout

class SingleEndExperiment(knock_knock.experiment.Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.outcome_fn_keys = ['outcome_list']

        self.diagram_kwargs.update(draw_sequence=True,
                                   flip_target=self.target_info.sequencing_direction == '-',
                                   highlight_SNPs=True,
                                   center_on_primers=True,
                                   split_at_indels=True,
                                  )
                                  
        # Remove any annotated HAs and replace them with inferred HAs.
        features_to_show = self.diagram_kwargs['features_to_show']

        for seq_name, name in sorted(features_to_show):
            if name.startswith('HA'):
                features_to_show.remove((seq_name, name))
                
        if (self.target_info.donor, self.target_info.donor_specific) in features_to_show:
            features_to_show.remove((self.target_info.donor, self.target_info.donor_specific))
            
        if self.target_info.inferred_HA_features is not None:
            for k in self.target_info.inferred_HA_features:
                features_to_show.add(k)

        self.x_tick_multiple = 100
        self.length_plot_smooth_window = 0

        self.read_types = [
            'trimmed',
            'trimmed_by_name',
            'nonredundant',
            'R1_no_overlap',
            'R2_no_overlap',
        ]

        self.layout_mode = 'amplicon'

    @memoized_property
    def max_relevant_length(self):
        return 600

    def __repr__(self):
        return f'SingleEndExperiment: group={self.group}, sample_name={self.sample_name}, base_dir={self.base_dir}'

    @property
    def preprocessed_read_type(self):
        return 'trimmed_by_name'

    @property
    def default_read_type(self):
        return 'trimmed_by_name'

    @property
    def read_types_to_align(self):
        return ['nonredundant']

    @memoized_property
    def categorizer(self):
        return repair_seq.pooled_layout.Layout
        #return repair_seq.prime_editing_layout.Layout

    def trim_reads(self):
        trimmed_fn = self.fns_by_read_type['fastq']['trimmed']

        try:
            fastq_fn = self.data_dir / self.description['fastq_fn']
        except KeyError:
            fastq_fn = self.data_dir / self.description['R1']

        try:
            reads = fastq.reads(fastq_fn, standardize_names=True)
        except OSError:
            print('error with', fastq_fn)
            raise

        with gzip.open(trimmed_fn, 'wt', compresslevel=1) as trimmed_fh:
            for read in self.progress(reads, desc='Trimming reads'):
                end = adapters.trim_by_local_alignment(adapters.truseq_R2_rc, read.seq)
                trimmed_fh.write(str(read[:end]))

    def sort_trimmed_reads(self):
        reads = sorted(self.reads_by_type('trimmed'), key=lambda read: read.name)
        fn = self.fns_by_read_type['fastq']['trimmed_by_name']
        with gzip.open(fn, 'wt', compresslevel=1) as sorted_fh:
            for read in reads:
                sorted_fh.write(str(read))

    def preprocess(self):
        self.trim_reads()
        self.sort_trimmed_reads()

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

                self.generate_read_lengths()
                self.generate_outcome_counts()

                self.record_sanitized_category_names()
            
            elif stage == 'visualize':
                self.generate_figures()
        except:
            print(self.group, self.sample_name)
            raise
