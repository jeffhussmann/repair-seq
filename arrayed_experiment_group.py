import gzip
import itertools
import time
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pysam
import yaml

import knock_knock.experiment
import knock_knock.visualize
import ddr.experiment_group

import ddr.prime_editing_experiment
import ddr.endogenous
import ddr.paired_end_experiment

from hits import utilities, fastq, sam

memoized_property = utilities.memoized_property

class Batch:
    def __init__(self, base_dir, batch):
        pass        

class ArrayedExperimentGroup(ddr.experiment_group.ExperimentGroup):
    def __init__(self, base_dir, batch, group, progress=None):
        self.base_dir = Path(base_dir)
        self.batch = batch
        self.group = group

        self.group_args = (base_dir, batch, group)

        super().__init__()

        self.progress = progress

        self.batch_sample_sheet_fn = self.data_dir / 'sample_sheet.csv'
        self.batch_group_descriptions_fn = self.data_dir / 'group_descriptions.csv'

        self.batch_sample_sheet = pd.read_csv(self.batch_sample_sheet_fn, index_col='sample_name')
        self.sample_sheet = self.batch_sample_sheet.query('group == @self.group')

        self.batch_group_descriptions = pd.read_csv(self.batch_group_descriptions_fn, index_col='group')
        self.description = self.batch_group_descriptions.loc[self.group]

        self.experiment_type = self.description['experiment_type']

        self.ExperimentType = arrayed_classes[self.experiment_type]
        self.CommonSequencesExperimentType = arrayed_classes[self.experiment_type, 'common_sequences']

        self.outcome_index_levels = ('category', 'subcategory', 'details')
        self.outcome_column_levels = ('condition', 'replicate')

    def __repr__(self):
        return f'ArrayedExperimentGroup: batch={self.batch}, group={self.group}, base_dir={self.base_dir}'

    @memoized_property
    def data_dir(self):
        return self.base_dir / 'data' / self.batch

    @memoized_property
    def results_dir(self):
        return self.base_dir / 'results' / self.batch / self.group

    def experiments(self, no_progress=False):
        for sample_name, row in self.sample_sheet.iterrows():
            yield self.sample_name_to_experiment(sample_name, no_progress=no_progress)

    @property
    def first_experiment(self):
        return next(self.experiments())

    @property
    def preprocessed_read_type(self):
        return self.first_experiment.preprocessed_read_type

    @property
    def categorizer(self):
        return self.first_experiment.categorizer

    @property
    def layout_mode(self):
        return self.first_experiment.layout_mode

    @property
    def target_info(self):
        return self.first_experiment.target_info

    @property
    def diagram_kwargs(self):
        return self.first_experiment.diagram_kwargs

    def common_sequence_chunk_exp_from_name(self, chunk_name):
        chunk_exp = self.CommonSequencesExperimentType(self.base_dir, self.batch, self.group, chunk_name,
                                        experiment_group=self,
                                        description=self.description,
                                       )
        return chunk_exp

    @memoized_property
    def num_experiments(self):
        return len(self.sample_sheet)

    @memoized_property
    def conditions(self):
        return sorted(self.sample_sheet['condition'].unique())

    @memoized_property
    def full_conditions(self):
        return list(zip(self.sample_sheet['condition'], self.sample_sheet['replicate']))

    @memoized_property
    def sample_names(self):
        return sorted(self.sample_sheet.index)

    def condition_replicates(self, condition):
        rows = self.sample_sheet.query('condition == @condition')
        return [self.sample_name_to_experiment(sample_name) for sample_name in rows.index]

    def sample_name_to_experiment(self, sample_name, no_progress=False):
        if no_progress:
            progress = None
        else:
            progress = self.progress

        exp = self.ExperimentType(self.base_dir, self.batch, self.group, sample_name, experiment_group=self, progress=progress)
        return exp

    @memoized_property
    def full_condition_to_experiment(self):
        full_condition_to_experiment = {}

        for exp in self.experiments():
            full_condition = exp.description['condition'], exp.description['replicate']
            full_condition_to_experiment[full_condition] = exp

        return full_condition_to_experiment

    def extract_genomic_insertion_length_distributions(self):
        length_distributions = {}
        
        for condition, exp in self.progress(self.full_condition_to_experiment.items()):
            for organism in ['hg19', 'bosTau7']:
                key = (*condition, organism)
                length_distributions[key] = np.zeros(1600)

            for outcome in exp.outcome_iter():
                if outcome.category == 'genomic insertion':
                    organism = outcome.subcategory
                    
                    lti  = ddr.outcome.LongTemplatedInsertionOutcome.from_string(outcome.details)
                    key = (*condition, organism)
                    length_distributions[key][lti.insertion_length()] += 1

        length_distributions_df = pd.DataFrame(length_distributions).T

        length_distributions_df.index.names = list(self.outcome_column_levels) + ['organism']

        # Normalize to number of valid reads in each sample.
        length_distributions_df = length_distributions_df.div(self.total_valid_reads, axis=0)

        length_distributions_df = length_distributions_df.reorder_levels(['organism'] + list(self.outcome_column_levels))

        length_distributions_df.to_csv(self.fns['genomic_insertion_length_distributions'])

    @memoized_property
    def genomic_insertion_length_distributions(self):
        num_index_cols = len(self.outcome_column_levels) + 1
        df = pd.read_csv(self.fns['genomic_insertion_length_distributions'], index_col=list(range(num_index_cols)))
        df.columns = [int(c) for c in df.columns]
        return df

    @memoized_property
    def outcome_counts(self):
        # Ignore nonspecific amplification products in denominator of any outcome fraction calculations.
        return self.outcome_counts_df(False).drop('nonspecific amplification', errors='ignore')

    @memoized_property
    def total_valid_reads(self):
        return self.outcome_counts.sum()

    @memoized_property
    def outcome_fractions(self):
        return self.outcome_counts / self.total_valid_reads

class ArrayedExperiment:
    def __init__(self, base_dir, batch, group, sample_name, experiment_group=None):
        if experiment_group is None:
            experiment_group = ArrayedExperimentGroup(base_dir, batch, group, type(self))

        self.base_dir = Path(base_dir)
        self.batch = batch
        self.group = group
        self.sample_name = sample_name
        self.experiment_group = experiment_group

    @property
    def default_read_type(self):
        # None required to trigger check for common sequence in alignment_groups
        return None

    def load_description(self):
        description = self.experiment_group.sample_sheet.loc[self.sample_name].to_dict()
        for key, value in self.experiment_group.description.items():
            description[key] = value
        return description

    @memoized_property
    def data_dir(self):
        return self.experiment_group.data_dir

    def make_nonredundant_sequence_fastq(self):
        # Extract reads with sequences that weren't seen more than once across the group.
        fn = self.fns_by_read_type['fastq']['nonredundant']
        with gzip.open(fn, 'wt', compresslevel=1) as fh:
            for read in self.reads_by_type(self.preprocessed_read_type):
                if read.seq not in self.experiment_group.common_sequence_to_outcome:
                    fh.write(str(read))

    @memoized_property
    def results_dir(self):
        return self.experiment_group.results_dir / self.sample_name

    @memoized_property
    def seq_to_outcome(self):
        seq_to_outcome = self.experiment_group.common_sequence_to_outcome
        for seq, outcome in seq_to_outcome.items():
            outcome.special_alignment = self.experiment_group.common_name_to_special_alignment.get(outcome.query_name)
        return seq_to_outcome

    @memoized_property
    def seq_to_alignments(self):
        return self.experiment_group.common_sequence_to_alignments

    def alignment_groups(self, fn_key='bam_by_name', outcome=None, read_type=None):
        if read_type is None:
            nonredundant_alignment_groups = super().alignment_groups(read_type='nonredundant', outcome=outcome)
            reads = self.reads_by_type(self.preprocessed_read_type)

            if outcome is None:
                outcome_records = itertools.repeat(None)
            else:
                #outcome_records = self.outcome_iter(outcome_fn_keys=['outcome_list'])
                outcome_records = self.outcome_iter()

            for read, outcome_record in zip(reads, outcome_records):
                if outcome is None or outcome_record.category == outcome or (outcome_record.category, outcome_record.subcategory) == outcome:
                    if read.seq in self.seq_to_alignments:
                        name = read.name
                        als = self.seq_to_alignments[read.seq]

                    else:
                        name, als = next(nonredundant_alignment_groups)

                        if name != read.name:
                            raise ValueError('iters out of sync', name, read.name)

                    yield name, als
        else:
            yield from super().alignment_groups(fn_key=fn_key, outcome=outcome, read_type=read_type)

    def categorize_outcomes(self, max_reads=None):
        # Record how long each categorization takes.
        times_taken = []

        if self.fns['outcomes_dir'].is_dir():
            shutil.rmtree(str(self.fns['outcomes_dir']))

        self.fns['outcomes_dir'].mkdir()

        outcome_to_qnames = defaultdict(list)

        bam_read_type = 'nonredundant'

        # iter wrap since tqdm objects are not iterators
        alignment_groups = iter(self.alignment_groups())

        if max_reads is not None:
            alignment_groups = itertools.islice(alignment_groups, max_reads)

        special_als = defaultdict(list)

        with self.fns['outcome_list'].open('w') as outcome_fh:

            for name, als in self.progress(alignment_groups, desc='Categorizing reads'):
                seq = als[0].get_forward_sequence()
                if seq in self.seq_to_outcome:
                    layout = self.seq_to_outcome[seq]
                    layout.query_name = name

                else:
                    layout = self.categorizer(als, self.target_info, mode=self.layout_mode)

                    try:
                        layout.categorize()
                    except:
                        print()
                        print(self.sample_name, name)
                        raise
                
                if layout.special_alignment is not None:
                    special_als[layout.category, layout.subcategory].append(layout.special_alignment)

                outcome_to_qnames[layout.category, layout.subcategory].append(name)

                outcome = self.final_Outcome.from_layout(layout)

                outcome_fh.write(f'{outcome}\n')

                times_taken.append(time.monotonic())

        # To make plotting easier, for each outcome, make a file listing all of
        # qnames for the outcome and a bam file (sorted by name) with all of the
        # alignments for these qnames.

        qname_to_outcome = {}

        bam_fn = self.fns_by_read_type['bam_by_name'][bam_read_type]
        header = sam.get_header(bam_fn)

        alignment_sorters = sam.multiple_AlignmentSorters(header, by_name=True)

        for outcome, qnames in outcome_to_qnames.items():
            outcome_fns = self.outcome_fns(outcome)
            outcome_fns['dir'].mkdir()

            alignment_sorters[outcome] = outcome_fns['bam_by_name'][bam_read_type]
            
            with outcome_fns['query_names'].open('w') as fh:
                for qname in qnames:
                    qname_to_outcome[qname] = outcome
                    fh.write(qname + '\n')
            
        with alignment_sorters:
            with pysam.AlignmentFile(bam_fn) as full_bam_fh:
                for al in self.progress(full_bam_fh, desc='Making outcome-specific bams'):
                    if al.query_name in qname_to_outcome:
                        outcome = qname_to_outcome[al.query_name]
                        alignment_sorters[outcome].write(al)

        # Make special alignments bams.
        for outcome, als in self.progress(special_als.items(), desc='Making special alignments bams'):
            outcome_fns = self.outcome_fns(outcome)
            bam_fn = outcome_fns['special_alignments']
            sorter = sam.AlignmentSorter(bam_fn, header)
            with sorter:
                for al in als:
                    sorter.write(al)

        return np.array(times_taken)

    @memoized_property
    def outcome_counts(self):
        counts = pd.read_csv(self.fns['outcome_counts'],
                             header=None,
                             index_col=[0, 1, 2],
                             squeeze=True,
                             na_filter=False,
                             sep='\t',
                            )
        counts.index.names = ['category', 'subcategory', 'details']
        return counts

    def load_outcome_counts(self):
        return self.outcome_counts.sum(level=[0, 1])

arrayed_classes = {}

for name, SpecializeExperiment in [('endogenous', ddr.endogenous.Experiment),
                                   ('paired_end', ddr.paired_end_experiment.PairedEndExperiment),
                                   ('prime_editing', ddr.prime_editing_experiment.PrimeEditingExperiment),
                                  ]:

        class ArrayedSpecializedExperiment(ArrayedExperiment, SpecializeExperiment):
            def __init__(self, base_dir, batch, group, sample_name, experiment_group=None, **kwargs):
                ArrayedExperiment.__init__(self, base_dir, batch, group, sample_name, experiment_group=experiment_group)
                SpecializeExperiment.__init__(self, base_dir, (batch, group), sample_name, **kwargs)
        
        arrayed_classes[name] = ArrayedSpecializedExperiment

        class ArrayedSpecializedCommonSequencesExperiment(ddr.experiment_group.CommonSequencesExperiment, ArrayedExperiment, SpecializeExperiment):
            def __init__(self, base_dir, batch, group, sample_name, experiment_group=None, **kwargs):
                ddr.experiment_group.CommonSequencesExperiment.__init__(self)
                ArrayedExperiment.__init__(self, base_dir, batch, group, sample_name, experiment_group=experiment_group)
                SpecializeExperiment.__init__(self, base_dir, (batch, group), sample_name, **kwargs)
        
        arrayed_classes[name, 'common_sequences'] = ArrayedSpecializedCommonSequencesExperiment
