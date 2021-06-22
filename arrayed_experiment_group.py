import gzip
import itertools
import time
import shutil
import warnings
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import pysam

from ipywidgets import Layout, Select

import ddr.experiment_group

import ddr.prime_editing_experiment
import ddr.endogenous
import ddr.paired_end_experiment
import ddr.single_end_experiment

from hits import utilities, sam
import knock_knock.explore

memoized_property = utilities.memoized_property

class Batch:
    def __init__(self, base_dir, batch,
                 category_groupings=None,
                 baseline_condition=None,
                 add_pseudocount=False,
                 only_edited=False,
                 progress=None,
                ):
        self.base_dir = Path(base_dir)
        self.batch = batch
        self.data_dir = self.base_dir / 'data' / batch

        if progress is None or getattr(progress, '_silent', False):
            def ignore_kwargs(x, **kwargs):
                return x
            progress = ignore_kwargs

        self.progress = progress

        self.category_groupings = category_groupings
        self.baseline_condition = baseline_condition
        self.add_pseudocount = add_pseudocount
        self.only_edited = only_edited

        self.sample_sheet_fn = self.data_dir / 'sample_sheet.csv'
        self.sample_sheet = pd.read_csv(self.sample_sheet_fn, index_col='sample_name')

        self.group_descriptions_fn = self.data_dir / 'group_descriptions.csv'
        self.group_descriptions = pd.read_csv(self.group_descriptions_fn, index_col='group')

        self.condition_colors_fn = self.data_dir / 'condition_colors.csv'
        if self.condition_colors_fn.exists():
            self.condition_colors = pd.read_csv(self.condition_colors_fn, index_col='perturbation', squeeze=True)
        else:
            self.condition_colors = None

    def __repr__(self):
        return f'Batch: {self.batch}, base_dir={self.base_dir}'

    @property
    def group_names(self):
        return self.sample_sheet['group'].unique()

    def group(self, group_name):
        return ArrayedExperimentGroup(self.base_dir, self.batch, group_name,
                                      category_groupings=self.category_groupings,
                                      baseline_condition=self.baseline_condition,
                                      add_pseudocount=self.add_pseudocount,
                                      only_edited=self.only_edited,
                                      progress=self.progress,
                                     )

    @memoized_property
    def groups(self):
        groups = {group_name: self.group(group_name) for group_name in self.group_names}
        return groups

    def group_query(self, query_string):
        groups = []

        for group_name, row in self.group_descriptions.query(query_string).iterrows():
            groups.append(self.groups[group_name])

        return groups

    def experiment_query(self, query_string):
        exps = []

        for sample_name, row in self.sample_sheet.query(query_string).iterrows():
            group = self.groups[row['group']]
            exp = group.sample_name_to_experiment(sample_name)
            exps.append(exp)

        return exps

def get_batch(base_dir, batch_name, progress=None, **kwargs):
    batch = None

    group_dir = Path(base_dir) / 'data' / batch_name
    group_descriptions_fn = group_dir / 'group_descriptions.csv'

    if group_descriptions_fn.exists():
        batch = Batch(base_dir, batch_name, progress, **kwargs)

    return batch

def get_all_batches(base_dir=Path.home() / 'projects' / 'ddr', progress=None, **kwargs):
    possible_batch_dirs = [p for p in (Path(base_dir) / 'results').iterdir() if p.is_dir()]

    batches = {}

    for possible_batch_dir in possible_batch_dirs:
        batch_name = possible_batch_dir.name
        batch = get_batch(base_dir, batch_name, progress=progress, **kwargs)
        if batch is not None:
            batches[batch_name] = batch

    return batches

class ArrayedExperimentGroup(ddr.experiment_group.ExperimentGroup):
    def __init__(self, base_dir, batch, group,
                 category_groupings=None,
                 progress=None,
                 baseline_condition=None,
                 add_pseudocount=None,
                 only_edited=False,
                ):
        self.base_dir = Path(base_dir)
        self.batch = batch
        self.group = group

        self.category_groupings = category_groupings
        self.add_pseudocount = add_pseudocount
        self.only_edited = only_edited

        self.group_args = (base_dir, batch, group)

        super().__init__()

        if progress is None or getattr(progress, '_silent', False):
            def ignore_kwargs(x, **kwargs):
                return x
            progress = ignore_kwargs

        self.silent = True

        self.progress = progress

        self.Batch = Batch(self.base_dir, self.batch)

        self.batch_sample_sheet = self.Batch.sample_sheet
        self.sample_sheet = self.batch_sample_sheet.query('group == @self.group').copy()

        self.description = self.Batch.group_descriptions.loc[self.group].copy()

        self.condition_keys = self.description['condition_keys'].split(';')
        self.full_condition_keys = tuple(self.condition_keys + ['replicate'])

        if baseline_condition is not None:
            self.baseline_condition = baseline_condition
        else:
            self.baseline_condition = tuple(self.description['baseline_condition'].split(';'))

        self.experiment_type = self.description['experiment_type']

        self.ExperimentType, self.CommonSequencesExperimentType = arrayed_specialized_experiment_factory(self.experiment_type)

        self.outcome_index_levels = ('category', 'subcategory', 'details')
        self.outcome_column_levels = self.full_condition_keys

        def condition_from_row(row):
            condition = tuple(row[key] for key in self.condition_keys)
            if len(condition) == 1:
                condition = condition[0]
            return condition

        def full_condition_from_row(row):
            return tuple(row[key] for key in self.full_condition_keys)

        self.full_conditions = [full_condition_from_row(row) for _, row in self.sample_sheet.iterrows()]

        conditions_are_unique = len(set(self.full_conditions)) == len(self.full_conditions)
        if not conditions_are_unique:
            print(f'{self}\nconditions are not unique:')
            for k, v in Counter(self.full_conditions).most_common():
                print(k, v)
            raise ValueError

        self.full_condition_to_sample_name = {full_condition_from_row(row): sample_name for sample_name, row in self.sample_sheet.iterrows()}

        self.conditions = sorted(set(c[:-1] for c in self.full_conditions))

        # Indexing breaks if it is a length 1 tuple.
        if len(self.condition_keys) == 1:
            self.baseline_condition = self.baseline_condition[0]
            self.conditions = [c[0] for c in self.conditions]

        self.sample_names = sorted(self.sample_sheet.index)

        self.condition_to_sample_names = defaultdict(list)
        for sample_name, row in self.sample_sheet.iterrows():
            condition = condition_from_row(row)
            self.condition_to_sample_names[condition].append(sample_name)

    def __repr__(self):
        return f'ArrayedExperimentGroup: batch={self.batch}, group={self.group}, base_dir={self.base_dir}'

    @memoized_property
    def data_dir(self):
        return self.base_dir / 'data' / self.batch

    @memoized_property
    def results_dir(self):
        return self.base_dir / 'results' / self.batch / self.group

    def experiments(self, no_progress=False):
        for sample_name in self.sample_names:
            yield self.sample_name_to_experiment(sample_name, no_progress=no_progress)

    @memoized_property
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

    def condition_replicates(self, condition):
        sample_names = self.condition_to_sample_names[condition]
        return [self.sample_name_to_experiment(sample_name) for sample_name in sample_names]

    def sample_name_to_experiment(self, sample_name, no_progress=False):
        if no_progress:
            progress = None
        else:
            progress = self.progress

        exp = self.ExperimentType(self.base_dir, self.batch, self.group, sample_name, experiment_group=self, progress=progress)
        return exp

    @memoized_property
    def full_condition_to_experiment(self):
        return {full_condition: self.sample_name_to_experiment(sample_name) for full_condition, sample_name in self.full_condition_to_sample_name.items()}

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
        to_drop = ['nonspecific amplification', 'bad sequence']

        # Empirically, overall editing rates can vary considerably across arrayed 
        # experiments, presumably due to nucleofection efficiency. If self.only_edited
        # is true, exlcude unedited reads from outcome counting.
        if self.only_edited:
            to_drop.append('wild type')

        outcome_counts = self.outcome_counts_df(False).drop(to_drop, errors='ignore')
        
        # Sort columns to avoid annoying pandas PerformanceWarnings.
        outcome_counts = outcome_counts.sort_index(axis='columns')

        return outcome_counts

    @memoized_property
    def outcome_counts_with_bad(self):
        outcome_counts = self.outcome_counts_df(False)
        
        # Sort columns to avoid annoying pandas PerformanceWarnings.
        outcome_counts = outcome_counts.sort_index(axis='columns')

        return outcome_counts

    @memoized_property
    def total_valid_reads(self):
        return self.outcome_counts.sum()

    @memoized_property
    def outcome_fractions(self):
        fractions = self.outcome_counts / self.total_valid_reads
        order = fractions[self.baseline_condition].mean(axis='columns').sort_values(ascending=False).index
        fractions = fractions.loc[order]
        return fractions

    @memoized_property
    def outcome_fractions_with_bad(self):
        return self.outcome_counts / self.outcome_counts.sum()

    @memoized_property
    def outcome_fraction_condition_means(self):
        return self.outcome_fractions.mean(axis='columns', level=self.condition_keys)

    @memoized_property
    def outcome_fraction_baseline_means(self):
        return self.outcome_fraction_condition_means[self.baseline_condition]

    @memoized_property
    def outcome_fraction_condition_stds(self):
        return self.outcome_fractions.std(axis='columns', level=self.condition_keys)

    @memoized_property
    def outcomes_by_baseline_frequency(self):
        return self.outcome_fraction_baseline_means.sort_values(ascending=False).index.values

    @memoized_property
    def outcome_fraction_differences(self):
        return self.outcome_fractions.sub(self.outcome_fraction_baseline_means, axis=0)

    @memoized_property
    def outcome_fraction_difference_condition_means(self):
        return self.outcome_fraction_differences.mean(axis='columns', level=self.condition_keys)

    @memoized_property
    def outcome_fraction_difference_condition_stds(self):
        return self.outcome_fraction_differences.std(axis='columns', level=self.condition_keys)

    @memoized_property
    def log2_fold_changes(self):
        # Using the warnings context manager doesn't work here, maybe because of pandas multithreading?
        warnings.filterwarnings('ignore')

        fold_changes = self.outcome_fractions.div(self.outcome_fraction_baseline_means, axis=0)
        log2_fold_changes = np.log2(fold_changes)

        warnings.resetwarnings()

        return log2_fold_changes

    @memoized_property
    def log2_fold_change_condition_means(self):
        return self.log2_fold_changes.mean(axis='columns', level=self.condition_keys)

    @memoized_property
    def log2_fold_change_condition_stds(self):
        return self.log2_fold_changes.std(axis='columns', level=self.condition_keys)

    @memoized_property
    def category_fractions(self):
        fs = self.outcome_fractions.sum(level='category')

        if self.category_groupings is not None:
            only_relevant_cats = pd.Index.difference(fs.index, self.category_groupings['not_relevant'])
            relevant_but_not_specific_cats = pd.Index.difference(only_relevant_cats, self.category_groupings['specific'])

            only_relevant = fs.loc[only_relevant_cats]

            only_relevant_normalized = only_relevant / only_relevant.sum()

            relevant_but_not_specific = only_relevant_normalized.loc[relevant_but_not_specific_cats].sum()

            grouped = only_relevant_normalized.loc[self.category_groupings['specific']]
            grouped.loc['all others'] = relevant_but_not_specific

            fs = grouped

            if self.add_pseudocount:
                reads_per_sample = self.outcome_counts.drop(self.category_groupings['not_relevant'], errors='ignore').sum()
                counts = fs * reads_per_sample
                counts += 1
                fs = counts / counts.sum()

        return fs

    @memoized_property
    def category_fraction_condition_means(self):
        return self.category_fractions.mean(axis='columns', level=self.condition_keys)

    @memoized_property
    def category_fraction_baseline_means(self):
        return self.category_fraction_condition_means[self.baseline_condition]

    @memoized_property
    def category_fraction_condition_stds(self):
        return self.category_fractions.std(axis='columns', level=self.condition_keys)

    @memoized_property
    def categories_by_baseline_frequency(self):
        return self.category_fraction_baseline_means.sort_values(ascending=False).index.values

    @memoized_property
    def category_fraction_differences(self):
        return self.category_fractions.sub(self.category_fraction_baseline_means, axis=0)

    @memoized_property
    def category_fraction_difference_condition_means(self):
        return self.category_fraction_differences.mean(axis='columns', level=self.condition_keys)

    @memoized_property
    def category_fraction_difference_condition_stds(self):
        return self.category_fraction_differences.std(axis='columns', level=self.condition_keys)

    @memoized_property
    def category_log2_fold_changes(self):
        # Using the warnings context manager doesn't work here, maybe because of pandas multithreading?
        warnings.filterwarnings('ignore')

        fold_changes = self.category_fractions.div(self.category_fraction_baseline_means, axis=0)
        log2_fold_changes = np.log2(fold_changes)

        warnings.resetwarnings()

        return log2_fold_changes

    @memoized_property
    def category_log2_fold_change_condition_means(self):
        # calculate mean in linear space, not log space
        fold_changes = self.category_fraction_condition_means.div(self.category_fraction_baseline_means, axis=0)
        return np.log2(fold_changes)

    @memoized_property
    def category_log2_fold_change_condition_stds(self):
        # calculate effective log2 fold change of mean +/- std in linear space
        means = self.category_fraction_condition_means
        stds = self.category_fraction_condition_stds
        baseline_means = self.category_fraction_baseline_means
        return {
            'lower': np.log2((means - stds).div(baseline_means, axis=0)),
            'upper': np.log2((means + stds).div(baseline_means, axis=0)),
        }

    # TODO: figure out how to avoid this hideous code duplication.

    @memoized_property
    def subcategory_fractions(self):
        return self.outcome_fractions.sum(level=['category', 'subcategory'])

    @memoized_property
    def subcategory_fraction_condition_means(self):
        return self.subcategory_fractions.mean(axis='columns', level=self.condition_keys)

    @memoized_property
    def subcategory_fraction_baseline_means(self):
        return self.subcategory_fraction_condition_means[self.baseline_condition]

    @memoized_property
    def subcategory_fraction_condition_stds(self):
        return self.subcategory_fractions.std(axis='columns', level=self.condition_keys)

    @memoized_property
    def subcategories_by_baseline_frequency(self):
        return self.subcategory_fraction_baseline_means.sort_values(ascending=False).index.values

    @memoized_property
    def subcategory_fraction_differences(self):
        return self.subcategory_fractions.sub(self.subcategory_fraction_baseline_means, axis=0)

    @memoized_property
    def subcategory_fraction_difference_condition_means(self):
        return self.subcategory_fraction_differences.mean(axis='columns', level=self.condition_keys)

    @memoized_property
    def subcategory_fraction_difference_condition_stds(self):
        return self.subcategory_fraction_differences.std(axis='columns', level=self.condition_keys)

    @memoized_property
    def subcategory_log2_fold_changes(self):
        # Using the warnings context manager doesn't work here, maybe because of pandas multithreading?
        warnings.filterwarnings('ignore')

        fold_changes = self.subcategory_fractions.div(self.subcategory_fraction_baseline_means, axis=0)
        log2_fold_changes = np.log2(fold_changes)

        warnings.resetwarnings()

        return log2_fold_changes

    @memoized_property
    def subcategory_log2_fold_change_condition_means(self):
        # calculate mean in linear space, not log space
        fold_changes = self.subcategory_fraction_condition_means.div(self.subcategory_fraction_baseline_means, axis=0)
        return np.log2(fold_changes)

    @memoized_property
    def subcategory_log2_fold_change_condition_stds(self):
        # calculate effective log2 fold change of mean +/- std in linear space
        means = self.subcategory_fraction_condition_means
        stds = self.subcategory_fraction_condition_stds
        baseline_means = self.subcategory_fraction_baseline_means
        return {
            'lower': np.log2((means - stds).div(baseline_means, axis=0)),
            'upper': np.log2((means + stds).div(baseline_means, axis=0)),
        }

    # Duplication of code in pooled_screen
    def donor_outcomes_containing_SNV(self, SNV_name):
        ti = self.target_info
        SNV_index = sorted(ti.donor_SNVs['target']).index(SNV_name)
        donor_base = ti.donor_SNVs['donor'][SNV_name]['base']
        nt_fracs = self.outcome_fraction_baseline_means
        outcomes = [(c, s, d) for c, s, d in nt_fracs.index.values if c == 'donor' and d[SNV_index] == donor_base]
        return outcomes

    @memoized_property
    def conversion_fractions(self):
        conversion_fractions = {}

        SNVs = self.target_info.donor_SNVs['target']

        outcome_fractions = self.outcome_fractions

        for SNV_name in SNVs:
            outcomes = self.donor_outcomes_containing_SNV(SNV_name)
            fractions = outcome_fractions.loc[outcomes].sum()
            conversion_fractions[SNV_name] = fractions

        conversion_fractions = pd.DataFrame.from_dict(conversion_fractions, orient='index').sort_index()
        
        return conversion_fractions

    def explore(self, **kwargs):
        explorer = ArrayedGroupExplorer(self, **kwargs)
        return explorer.layout

class ArrayedExperiment:
    def __init__(self, base_dir, batch, group, sample_name, experiment_group=None):
        if experiment_group is None:
            experiment_group = ArrayedExperimentGroup(base_dir, batch, group, type(self))

        self.base_dir = Path(base_dir)
        self.batch = batch
        self.group = group
        self.sample_name = sample_name
        self.experiment_group = experiment_group

        self.error_corrected = False

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

                # Special handling of empty sequence.
                if seq is None:
                    seq = ''

                if seq in self.seq_to_outcome:
                    layout = self.seq_to_outcome[seq]
                    layout.query_name = name

                else:
                    layout = self.categorizer(als, self.target_info, error_corrected=self.error_corrected, mode=self.layout_mode)

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

def arrayed_specialized_experiment_factory(experiment_kind):
    experiment_kind_to_class = {
        'paired_end': ddr.paired_end_experiment.PairedEndExperiment,
        'prime_editing': ddr.prime_editing_experiment.PrimeEditingExperiment,
        'single_end': ddr.single_end_experiment.SingleEndExperiment,
    }

    SpecializedExperiment = experiment_kind_to_class[experiment_kind]

    class ArrayedSpecializedExperiment(ArrayedExperiment, SpecializedExperiment):
        def __init__(self, base_dir, batch, group, sample_name, experiment_group=None, **kwargs):
            ArrayedExperiment.__init__(self, base_dir, batch, group, sample_name, experiment_group=experiment_group)
            SpecializedExperiment.__init__(self, base_dir, (batch, group), sample_name, **kwargs)

        def __repr__(self):
            return f'Arrayed{SpecializedExperiment.__repr__(self)}'
    
    class ArrayedSpecializedCommonSequencesExperiment(ddr.experiment_group.CommonSequencesExperiment, ArrayedExperiment, SpecializedExperiment):
        def __init__(self, base_dir, batch, group, sample_name, experiment_group=None, **kwargs):
            ddr.experiment_group.CommonSequencesExperiment.__init__(self)
            ArrayedExperiment.__init__(self, base_dir, batch, group, sample_name, experiment_group=experiment_group)
            SpecializedExperiment.__init__(self, base_dir, (batch, group), sample_name, **kwargs)
    
    return ArrayedSpecializedExperiment, ArrayedSpecializedCommonSequencesExperiment

class ArrayedGroupExplorer(knock_knock.explore.Explorer):
    def __init__(self,
                 group,
                 initial_condition=None,
                 by_outcome=True,
                 **plot_kwargs,
                ):
        self.group = group

        if initial_condition is None:
            initial_condition = self.group.conditions[0]
        self.initial_condition = initial_condition

        self.experiments = {}

        super().__init__(by_outcome, **plot_kwargs)

    def populate_replicates(self, change):
        with self.output:
            condition = self.widgets['condition'].value
            exps = self.group.condition_replicates(condition)

            self.widgets['replicate'].options = [(exp.description['replicate'], exp) for exp in exps]
            self.widgets['replicate'].index = 0

    def get_current_experiment(self):
        experiment = self.widgets['replicate'].value
        return experiment

    def set_up_read_selection_widgets(self):
        condition_options = [(', '.join(c), c) for c in self.group.conditions] 
        self.widgets.update({
            'condition': Select(options=condition_options, value=self.initial_condition, layout=Layout(height='200px', width='300px')),
            'replicate': Select(options=[], layout=Layout(height='200px', width='150px')),
        })

        self.populate_replicates({'name': 'initial'})
        self.widgets['condition'].observe(self.populate_replicates, names='value')

        if self.by_outcome:
            self.populate_categories({'name': 'initial'})
            self.populate_subcategories({'name': 'initial'})

            self.widgets['replicate'].observe(self.populate_categories, names='value')
            self.widgets['category'].observe(self.populate_subcategories, names='value')
            self.widgets['subcategory'].observe(self.populate_read_ids, names='value')
            selection_widget_keys = ['condition', 'replicate', 'category', 'subcategory', 'read_id']
        else:
            self.widgets['replicate'].observe(self.populate_read_ids, names='value')
            selection_widget_keys = ['condition', 'replicate', 'read_id']

        self.populate_read_ids({'name': 'initial'})

        return selection_widget_keys