import shutil
import os
import bisect
import pickle
import contextlib
import subprocess
import resource
import heapq
import gzip
import itertools
import time
import datetime
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import scipy.sparse
import pandas as pd
import numpy as np
import yaml
import ipywidgets
import pysam
import nbconvert
import nbformat
import h5py
import tqdm

from hits import utilities, sam, fastq, fasta, mapping_tools, annotation, interval
from knock_knock import experiment, target_info, visualize

from . import pooled_layout, collapse, coherence, guide_library, prime_editing_layout

memoized_property = utilities.memoized_property
memoized_with_key = utilities.memoized_with_key

ALL_NON_TARGETING = 'all_non_targeting'

class SingleGuideExperiment(experiment.Experiment):
    def __init__(self, base_dir, group, fixed_guide, variable_guide, **kwargs):
        name = f'{fixed_guide}-{variable_guide}'

        # Provide an option to pass in a premade pool to prevent excessive remaking.
        self.pool = kwargs.pop('pool', None)
        if self.pool is None:
            self.pool = PooledScreen(base_dir, group)

        self.fixed_guide = fixed_guide
        self.variable_guide = variable_guide
        
        super().__init__(base_dir, group, name, **kwargs)

        self.fns.update({
            'chunks': self.results_dir / 'chunks',
            'UMIs_seen': self.results_dir / 'UMIs_seen.txt',
            'guide_mismatch_rates': self.results_dir / 'guide_mismatch_rates.txt',
            'truncation_positions': self.results_dir / 'truncation_positions.txt',

            'genomic_insertion_seqs': self.results_dir / 'genomic_insertion_seqs.fa',
            'filtered_genomic_insertion_seqs': self.results_dir / 'filtered_genomic_insertion_seqs.fa',
            'filtered_genomic_insertion_details': self.results_dir / 'filtered_genomic_insertion_details.txt',

            'filtered_templated_insertion_details': self.results_dir / 'filtered_templated_insertion_details.hdf5',
            'filtered_duplication_details': self.results_dir / 'filtered_duplication_details.hdf5',

            'collapsed_UMI_outcomes': self.results_dir / 'collapsed_UMI_outcomes.txt',
            'cell_outcomes': self.results_dir / 'cell_outcomes.txt',
            'filtered_cell_outcomes': self.results_dir / 'filtered_cell_outcomes.txt',

            'filtered_cell_bam': self.results_dir / 'filtered_cell_aligments.bam',
            'reads_per_UMI': self.results_dir / 'reads_per_UMI.pkl',
        })

        self.max_insertion_length = None
        self.max_qual = 41

        self.min_reads_per_cluster = 2

        self.use_memoized_outcomes = kwargs.get('use_memoized_outcomes', True)

        self.read_types = [
            'collapsed_R2',
            'collapsed_uncommon_R2',
        ]

        self.supplemental_index_names = [
            'hg19',
            'bosTau7',
        ]

        self.layout_mode = 'error_corrected'

        self.min_reads_per_UMI = self.pool.min_reads_per_UMI

        self.outcome_fn_keys = ['filtered_cell_outcomes']

    @property
    def final_Outcome(self):
        return coherence.Pooled_UMI_Outcome

    def load_description(self):
        return self.pool.sample_sheet

    @memoized_property
    def categorizer(self):
        return self.pool.categorizer

    @memoized_property
    def has_UMIs(self):
        return self.pool.has_UMIs

    @memoized_property
    def target_name(self):
        prefix = self.description['target_info_prefix']

        if self.fixed_guide == 'none':
            if self.sample_name == 'unknown':
                target_name = prefix
            else:
                target_name = f'{prefix}_{self.pool.variable_guide_library.name}_{self.variable_guide}'
        else:
            if self.fixed_guide == 'unknown':
                target_name = prefix
            else:
                target_name = f'{prefix}-{self.fixed_guide}-{self.variable_guide}'

        return target_name

    @property
    def reads(self):
        # To merge many chunks, need to raise limit on open files.
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))

        chunks_dir = self.fns['chunks']
        chunk_fns = sorted(chunks_dir.glob(f'*R2.fastq.gz'))
        chunks = [fastq.reads(fn, up_to_space=True) for fn in chunk_fns]
        merged_reads = heapq.merge(*chunks, key=lambda r: r.name)

        return self.progress(merged_reads)

    def get_read_alignments(self, read_id, fn_key='bam_by_name', outcome=None, read_type=None):
        # Note: read_type is ignored but needed for function signature.
        looked_up_common = False

        if self.use_memoized_outcomes:
            seq = self.names_with_common_seq.get(read_id)
            if seq is not None:
                als = self.pool.get_common_seq_alignments(seq)
                looked_up_common = True
            read_type = 'collapsed_uncommon_R2'
        else:
            read_type = 'collapsed_R2'
            
        if not looked_up_common:
            als = super().get_read_alignments(read_id, fn_key=fn_key, outcome=outcome, read_type=read_type)

        return als

    def alignment_group_dictionary(self, category, subcategory, n=100):
        relevant_cells = self.filtered_cell_outcomes.query('category == @category and subcategory == @subcategory')
        sample = relevant_cells.sample(min(n, len(relevant_cells)), random_state=0)

        # qname column has different names in +/- UMI experiments
        if 'original_name' in sample.columns:
            qname_key = 'original_name'
        else:
            qname_key = 'standardized_qname'

        qnames = set(sample[qname_key])

        outcome_alignment_groups = self.alignment_groups(outcome=(category, subcategory), read_type='collapsed_uncommon_R2')
        name_with_uncommon_seq_to_als = dict(outcome_alignment_groups)

        name_to_als = {}
        for qname in qnames:
            als = name_with_uncommon_seq_to_als.get(qname)
            if als is None:
                seq = self.names_with_common_seq.get(qname)
                if seq is None:
                    raise ValueError(qname)
                else:
                    als = self.pool.get_common_seq_alignments(seq)

            name_to_als[qname] = als

        return name_to_als

    @memoized_property
    def R2_read_length(self):
        return len(next(iter(self.collapsed_reads(no_progress=True))))

    def preprocess_R2_reads(self):
        if self.has_UMIs:
            self.collapse_UMI_reads()
        else:
            self.merge_read_chunks()
    
    def collapse_UMI_reads(self):
        ''' Takes R2_fn sorted by UMI and collapses reads with the same UMI and
        sufficiently similar sequence.
        '''

        def UMI_key(read):
            return collapse.Annotations['UMI_guide'].from_identifier(read.name)['UMI']

        def num_reads_key(read):
            return collapse.Annotations['collapsed_UMI'].from_identifier(read.name)['num_reads']

        R1_read_length = 45

        mismatch_counts = np.zeros(R1_read_length)
        total = 0

        expected_seq = self.pool.variable_guide_library.guides_df.loc[self.variable_guide, 'full_seq'][:R1_read_length].upper()

        collapsed_fn = self.fns_by_read_type['fastq']['collapsed_R2']

        UMIs_seen = defaultdict(list)

        with gzip.open(collapsed_fn, 'wt', compresslevel=1) as collapsed_fh:
            groups = utilities.group_by(self.reads, UMI_key)
            for UMI, UMI_group in groups:
                clusters = collapse.form_clusters(UMI_group, max_read_length=None, max_hq_mismatches=0)
                clusters = sorted(clusters, key=num_reads_key, reverse=True)

                for i, cluster in enumerate(clusters):
                    annotation = collapse.Annotations['collapsed_UMI'].from_identifier(cluster.name)
                    annotation['UMI'] = UMI
                    annotation['cluster_id'] = i

                    UMIs_seen[UMI].append(annotation['num_reads'])

                    if annotation['num_reads'] >= self.min_reads_per_cluster:
                        total += 1
                        guide = annotation['guide']
                        if guide == expected_seq:
                            mismatch = -1
                        else:
                            qs = fastq.decode_sanger(annotation['guide_qual'])
                            mismatches = []
                            for i, (seen, expected, q) in enumerate(zip(guide, expected_seq, qs)):
                                if seen != expected and q >= 30:
                                    mismatches.append(i)

                            if len(mismatches) == 0:
                                mismatch = -1
                            elif len(mismatches) == 1:
                                mismatch = mismatches[0]
                            elif len(mismatches) > 1:
                                continue

                            mismatch_counts[mismatch] += 1

                        mismatch_annotation = collapse.Annotations['collapsed_UMI_mismatch'](annotation)
                        mismatch_annotation['mismatch'] = mismatch

                        cluster.name = str(mismatch_annotation)

                        collapsed_fh.write(str(cluster))

        mismatch_rates = mismatch_counts / (max(total, 1))
        np.savetxt(self.fns['guide_mismatch_rates'], mismatch_rates)

        with open(self.fns['UMIs_seen'], 'w') as fh:
            for UMI in sorted(UMIs_seen):
                cluster_sizes = ','.join(str(size) for size in UMIs_seen[UMI])
                fh.write(f'{UMI}\t{cluster_sizes}\n')

    def collapsed_reads(self, no_progress=False):
        fn = self.fns_by_read_type['fastq']['collapsed_R2']
        reads = fastq.reads(fn)
        if no_progress:
            return reads
        else:
            return self.progress(reads)

    def make_uncommon_sequence_fastq(self):
        fn = self.fns_by_read_type['fastq']['collapsed_uncommon_R2']
        with gzip.open(fn, 'wt', compresslevel=1) as fh:
            for read in self.collapsed_reads():
                if read.seq not in self.pool.common_sequence_to_outcome:
                    fh.write(str(read))

    @memoized_property
    def names_with_common_seq(self):
        names = {}

        for read in self.collapsed_reads():
            if read.seq in self.pool.common_sequence_to_outcome:
                names[read.name] = read.seq

        return names

    @property
    def collapsed_uncommon_reads(self):
        fn = self.fns_by_read_type['fastq']['collapsed_uncommon_R2']
        return self.progress(fastq.reads(fn))

    @memoized_property
    def combined_header(self):
        return sam.get_header(self.fns['bam_by_name'])
        
    def categorize_outcomes(self, max_reads=None):
        times = []

        if self.fns['outcomes_dir'].is_dir():
            shutil.rmtree(str(self.fns['outcomes_dir']))

        self.fns['outcomes_dir'].mkdir()

        outcomes = defaultdict(list)

        total = 0
        required_sw = 0

        if self.use_memoized_outcomes:
            outcome_lookup = self.pool.common_sequence_to_outcome
            special_alignment_lookup = self.pool.common_sequence_to_special_alignment
            bam_read_type = 'collapsed_uncommon_R2'
        else:
            outcome_lookup = {}
            special_alignment_lookup = {}
            bam_read_type = 'collapsed_R2'

        # iter wrap since tqdm objects are not iterators
        alignment_groups = iter(self.alignment_groups(fn_key='bam_by_name', read_type=bam_read_type))
        reads = self.reads_by_type('collapsed_R2')

        if max_reads is not None:
            reads = itertools.islice(reads, max_reads)

        special_als = defaultdict(list)

        with self.fns['outcome_list'].open('w') as outcome_fh, \
             self.fns['genomic_insertion_seqs'].open('w') as genomic_insertion_seqs_fh:

            for read in self.progress(reads, desc='Categorizing reads'):
                if read.seq in outcome_lookup:
                    category, subcategory, details = outcome_lookup[read.seq]
                    special_alignment = special_alignment_lookup.get(read.seq)

                else:
                    name, als = next(alignment_groups)
                    if name != read.name:
                        raise ValueError('iters out of sync', name, read.name)

                    layout = self.categorizer(als, self.target_info, mode=self.layout_mode)
                    total += 1
                    try:
                        category, subcategory, details, outcome = layout.categorize()

                        if outcome is not None:
                            # Translate positions to be relative to a registered anchor
                            # on the target sequence.
                            details = str(outcome.perform_anchor_shift(self.target_info.anchor))

                    except:
                        print()
                        print(self.name, name)
                        raise
                
                    if layout.required_sw:
                        required_sw += 1

                    special_alignment = layout.special_alignment

                if special_alignment is not None:
                    special_als[category, subcategory].append(special_alignment)

                outcomes[category, subcategory].append(read.name)

                if self.has_UMIs or isinstance(self, CommonSequenceExperiment):
                    annotation = collapse.Annotations['collapsed_UMI_mismatch'].from_identifier(read.name)

                    if category in ['uncategorized', 'SD-MMEJ'] and not self.use_memoized_outcomes:
                        if int(annotation['UMI']) < 1000: 
                            details = '{},{}_{}'.format(details, annotation['UMI'], annotation['num_reads'])

                    outcome = coherence.Pooled_UMI_Outcome(annotation['UMI'],
                                                        annotation['mismatch'],
                                                        annotation['cluster_id'],
                                                        annotation['num_reads'],
                                                        category,
                                                        subcategory,
                                                        details,
                                                        read.name,
                                                        )
                else:
                    outcome = coherence.gDNA_Outcome(read.name,
                                                     category,
                                                     subcategory,
                                                     details,
                                                    )

                outcome_fh.write(str(outcome) + '\n')

                if category == 'genomic insertion' and subcategory == 'hg19':
                    cropped_genomic_alignment = special_alignment
                    query_bounds = interval.get_covered(cropped_genomic_alignment)

                    start = query_bounds.start
                    end = query_bounds.end
                    if end is not None:
                        end += 1

                    inserted_sequence = read.seq[start:end]
                    record = fasta.Record(read.name, inserted_sequence)
                    genomic_insertion_seqs_fh.write(str(record))
                
                times.append(time.monotonic())

        # To make plotting easier, for each outcome, make a file listing all of
        # qnames for the outcome and a bam file (sorted by name) with all of the
        # alignments for these qnames.

        qname_to_outcome = {}
        bam_fhs = {}

        bam_fn = self.fns_by_read_type['bam_by_name'][bam_read_type]
        with pysam.AlignmentFile(bam_fn) as full_bam_fh:
            header = full_bam_fh.header

        for outcome, qnames in outcomes.items():
            outcome_fns = self.outcome_fns(outcome)
            outcome_fns['dir'].mkdir()
            bam_fhs[outcome] = pysam.AlignmentFile(outcome_fns['bam_by_name'][bam_read_type], 'wb', header=header)
            
            with outcome_fns['query_names'].open('w') as fh:
                for qname in qnames:
                    qname_to_outcome[qname] = outcome
                    fh.write(qname + '\n')
            
        with pysam.AlignmentFile(bam_fn) as full_bam_fh:
            for al in self.progress(full_bam_fh, desc='Making outcome-specific bams'):
                if al.query_name in qname_to_outcome:
                    outcome = qname_to_outcome[al.query_name]
                    bam_fhs[outcome].write(al)

        for outcome, fh in bam_fhs.items():
            fh.close()

        # Make special alignments bams.
        for outcome, als in self.progress(special_als.items(), desc='Making special alignments bams'):
            outcome_fns = self.outcome_fns(outcome)
            bam_fn = outcome_fns['special_alignments']
            sorter = sam.AlignmentSorter(bam_fn, header)
            with sorter:
                for al in als:
                    sorter.write(al)

        return np.array(times)

    @memoized_property
    def outcome_counts(self):
        counts = pd.read_csv(self.fns['outcome_counts'],
                             header=None,
                             index_col=[0, 1, 2, 3],
                             squeeze=True,
                             na_filter=False,
                             sep='\t',
                            )
        counts.index.names = ['perfect_guide', 'category', 'subcategory', 'details']
        return counts

    @memoized_property
    def outcomes(self):
        ''' (category, subcategory) tuples in descending order by total count '''
        totals = self.outcome_counts[True].groupby(by=['category', 'subcategory']).sum().sort_values(ascending=False)
        return totals.index.values
    
    @memoized_property
    def perfect_guide_outcome_counts(self):
        return self.outcome_counts.xs(True)

    def collapse_UMI_outcomes(self):
        all_collapsed_outcomes, most_abundant_outcomes = coherence.collapse_pooled_UMI_outcomes(self.fns['outcome_list'])
        with self.fns['collapsed_UMI_outcomes'].open('w') as fh:
            for outcome in all_collapsed_outcomes:
                fh.write(str(outcome) + '\n')
        
        with self.fns['cell_outcomes'].open('w') as fh:
            for outcome in most_abundant_outcomes:
                fh.write(str(outcome) + '\n')
        
        counts = Counter()
        with self.fns['filtered_cell_outcomes'].open('w') as fh:
            for outcome in most_abundant_outcomes:
                if outcome.num_reads >= self.min_reads_per_UMI:
                    fh.write(str(outcome) + '\n')
                    perfect = (outcome.guide_mismatch == -1)
                    counts[perfect, outcome.category, outcome.subcategory, outcome.details] += 1

        counts = pd.Series(counts).sort_values(ascending=False)
        counts.to_csv(self.fns['outcome_counts'], sep='\t', header=False)
        
    def make_filtered_cell_bams(self):
        # Make bams containing only alignments from final cell assignments for IGV browsing.
        cells = self.filtered_cell_outcomes
        name_to_outcome = {}
        for _, row in cells.query('guide_mismatch == -1').iterrows():
            name_to_outcome[row['original_name']] = (row['category'], row['subcategory'])

        outcomes_seen = cells.groupby(by=['category', 'subcategory']).size().index.values

        # Because of common outcome memoization, outcome dirs may not exist for every outcome.
        for outcome in outcomes_seen:
            self.outcome_fns(outcome)['dir'].mkdir(exist_ok=True)

        if self.use_memoized_outcomes:
            bam_read_type = 'collapsed_uncommon_R2'
        else:
            bam_read_type = 'collapsed_R2'

        bam_fn = self.fns_by_read_type['bam_by_name'][bam_read_type]

        with pysam.AlignmentFile(bam_fn) as combined_bam_fh:
            sorters = sam.multiple_AlignmentSorters(combined_bam_fh.header)
            sorters['all'] = self.fns['filtered_cell_bam']

            for outcome in outcomes_seen:
                sorters[outcome] = self.outcome_fns(outcome)['filtered_cell_bam']

            with sorters:
                for alignment in self.progress(combined_bam_fh, desc='Making filtered cell bams'):
                    outcome = name_to_outcome.get(alignment.query_name)
                    if outcome is not None:
                        sorters['all'].write(alignment)
                        sorters[outcome].write(alignment)

        for outcome in outcomes_seen:
            in_fn = self.outcome_fns(outcome)['filtered_cell_bam']
            out_fn = self.outcome_fns(outcome)['filtered_cell_bam_by_name']
            sam.sort_bam(in_fn, out_fn, by_name=True)
    
    def make_reads_per_UMI(self, individual_outcomes=None):
        if individual_outcomes is None:
            individual_outcomes = set()

        reads_per_UMI = defaultdict(Counter)

        with open(self.fns['cell_outcomes']) as fh:
            for line in fh:
                outcome = coherence.Pooled_UMI_Outcome.from_line(line)

                reads_per_UMI['all'][outcome.num_reads] += 1

                reads_per_UMI[outcome.category, outcome.subcategory][outcome.num_reads] += 1

                if outcome.outcome in individual_outcomes:
                    reads_per_UMI[outcome.outcome][outcome.num_reads] += 1

        with open(str(self.fns['reads_per_UMI']), 'wb') as fh:
            pickle.dump(reads_per_UMI, fh)

    @memoized_property
    def reads_per_UMI(self):
        with open(str(self.fns['reads_per_UMI']), 'rb') as fh:
            reads_per_UMI = pickle.load(fh)
        return reads_per_UMI

    @memoized_property
    def cell_outcomes(self):
        df = pd.read_csv(self.fns['cell_outcomes'], header=None, na_filter=False, names=coherence.Pooled_UMI_Outcome.columns, sep='\t')
        return df

    @memoized_property
    def filtered_cell_outcomes(self):
        df = pd.read_csv(self.fns['filtered_cell_outcomes'], header=None, na_filter=False, names=coherence.Pooled_UMI_Outcome.columns, sep='\t')
        return df

    def example_diagrams(self, outcome, num_examples, **kwargs):
        category, subcategory = outcome
        alignment_group_dictionary = self.alignment_group_dictionary(category, subcategory, n=num_examples)
        alignment_groups = alignment_group_dictionary.items()
        diagrams = self.alignment_groups_to_diagrams(alignment_groups, num_examples, flip_target=True)
        return diagrams

    def length_distribution_figure(self, *args, **kwargs):
        return None

    def get_category_subcategory_pairs(self):
        pairs = self.filtered_cell_outcomes.groupby(by=['category', 'subcategory']).size().sort_values(ascending=False).index.values
        return pairs

    def make_all_diagram_notebooks(self):
        for category, subcategory in self.progress(self.get_category_subcategory_pairs()):
            self.make_diagram_notebook(category, subcategory)

    def get_read_layout(self, read_id, qname_to_als=None, outcome=None):
        if qname_to_als is None:
            als = self.get_read_alignments(read_id, outcome=outcome)
        else:
            als = qname_to_als[read_id]
        layout = self.categorizer(als, self.target_info, mode=self.layout_mode)
        return layout

    def get_read_diagram(self, read_id, only_relevant=True, qname_to_als=None, outcome=None, **diagram_kwargs):
        default_diagram_kwargs = dict(
            ref_centric=True,
            highlight_SNPs=True,
            flip_target=True,
            #target_on_top=True,
            split_at_indels=True,
            force_left_aligned=True,
            draw_sequence=True,
            draw_mismatches=True,    
            features_to_show=self.target_info.features_to_show,
            refs_to_hide={'ssODN_dummy', 'ssODN_Cpf1'},
            title='',
        )

        for k, v in default_diagram_kwargs.items():
            diagram_kwargs.setdefault(k, v)

        layout = self.get_read_layout(read_id, qname_to_als=qname_to_als, outcome=outcome)

        if only_relevant:
            layout.categorize()
            to_plot = layout.relevant_alignments
        else:
            to_plot = layout.alignments

        diagram = visualize.ReadDiagram(to_plot, self.target_info, **diagram_kwargs)

        return diagram

    def extract_truncation_positions(self):
        counts = np.zeros(len(self.target_info.target_sequence), int)

        with open(self.fns['filtered_cell_outcomes']) as fh:
            for line in fh:
                outcome = coherence.Pooled_UMI_Outcome.from_line(line)
                if outcome.category == 'truncation' and outcome.guide_mismatch == -1 and outcome.details != 'None':
                    counts[int(outcome.details)] += 1

        np.savetxt(self.fns['truncation_positions'], counts, fmt='%d')
        
    @memoized_property
    def truncation_positions(self):
        return np.loadtxt(self.fns['truncation_positions'], int)

    def extract_templated_insertion_info(self):
        fields = pooled_layout.LongTemplatedInsertionOutcome.int_fields
        
        lists = defaultdict(list)

        with open(self.fns['filtered_cell_outcomes']) as outcomes_fh:
            for line in outcomes_fh:
                outcome = self.final_Outcome.from_line(line)
            
                if outcome.category in ['donor insertion', 'genomic insertion', 'unintended donor integration'] and getattr(outcome, 'guide_mismatch', -1):
                    insertion_outcome = pooled_layout.LongTemplatedInsertionOutcome.from_string(outcome.details)
                    
                    for field in fields: 
                        value = getattr(insertion_outcome, field)
                        key = f'{outcome.category}/{outcome.subcategory}/{field}'
                        lists[key].append(value)

        with h5py.File(self.fns['filtered_templated_insertion_details'], 'w') as hdf5_file:
            cat_and_subcats = {key.rsplit('/', 1)[0] for key in lists}

            for cat_and_subcat in cat_and_subcats:
                left_key = f'{cat_and_subcat}/left_insertion_query_bound'
                right_key = f'{cat_and_subcat}/right_insertion_query_bound'

                lengths = []

                for left, right in zip(lists[left_key], lists[right_key]):
                    if right == self.R2_read_length - 1:
                        length = self.R2_read_length
                    else:
                        length = right - left + 1

                    lengths.append(length)

                lengths_key = f'{cat_and_subcat}/insertion_length'

                lists[lengths_key] = lengths
        
            for key, value_list in lists.items():
                hdf5_file.create_dataset(f'{key}/list', data=np.array(value_list))

                counts = Counter(value_list)

                if len(counts) == 0:
                    values = np.array([], dtype=int)
                    counts = np.array([], dtype=int)
                else:
                    values = np.array(sorted(counts))
                    counts = np.array([counts[v] for v in values])

                hdf5_file.create_dataset(f'{key}/values', data=values)
                hdf5_file.create_dataset(f'{key}/counts', data=counts)

    def extract_duplication_info(self):
        lists = defaultdict(list)

        with open(self.fns['filtered_cell_outcomes']) as outcomes_fh:
            for line in outcomes_fh:
                outcome = self.final_Outcome.from_line(line)
            
                if outcome.category in ['duplication'] and getattr(outcome, 'guide_mismatch', -1):
                    duplication_outcome = prime_editing_layout.DuplicationOutcome.from_string(outcome.details)
                    
                    for side, value in zip(['left', 'right'], duplication_outcome.ref_junctions[0]):
                        key = f'{outcome.category}/{outcome.subcategory}/junction_{side}_ref_edge'
                        lists[key].append(value)

        with h5py.File(self.fns['filtered_duplication_details'], 'w') as hdf5_file:
            cat_and_subcats = {key.rsplit('/', 1)[0] for key in lists}

            for key, value_list in lists.items():
                hdf5_file.create_dataset(f'{key}/list', data=np.array(value_list))

                counts = Counter(value_list)

                if len(counts) == 0:
                    values = np.array([], dtype=int)
                    counts = np.array([], dtype=int)
                else:
                    values = np.array(sorted(counts))
                    counts = np.array([counts[v] for v in values])

                hdf5_file.create_dataset(f'{key}/values', data=values)
                hdf5_file.create_dataset(f'{key}/counts', data=counts)

    @memoized_property
    def donor_misintegration_info(self):
        outcomes = []
        with open(self.fns['filtered_donor_misintegration_details']) as details_fh:
            for line in details_fh:
                outcome = pooled_layout.LongTemplatedInsertionOutcome.from_string(line.strip())
                outcomes.append(outcome)
        return outcomes

    @memoized_property
    def donor_pastein_info(self):
        outcomes = []
        with open(self.fns['filtered_donor_paste-in_details']) as details_fh:
            for line in details_fh:
                outcome = pooled_layout.LongTemplatedInsertionOutcome.from_string(line.strip())
                outcomes.append(outcome)
        return outcomes

    def extract_genomic_insertion_info(self):
        genomic_insertion_seqs = fasta.to_dict(self.fns['genomic_insertion_seqs'])

        with open(self.fns['outcome_list']) as outcomes_fh, \
             open(self.fns['filtered_genomic_insertion_seqs'], 'w') as seqs_fh, \
             open(self.fns['filtered_genomic_insertion_details'], 'w') as details_fh:

            for line in outcomes_fh:
                outcome = coherence.Pooled_UMI_Outcome.from_line(line)
                if outcome.category == 'genomic insertion' and outcome.subcategory == 'hg19':
                    name = outcome.original_name
                    record = fasta.Record(name, genomic_insertion_seqs[name])
                    seqs_fh.write(str(record))

                    details_fh.write(f'{outcome.details}\n')

    @memoized_property
    def genomic_insertion_info(self):
        outcomes = []
        with open(self.fns['filtered_genomic_insertion_details']) as details_fh:
            for line in details_fh:
                outcome = pooled_layout.LongTemplatedInsertionOutcome.from_string(line.strip())
                outcomes.append(outcome)
        return outcomes

    def process(self, stage):
        try:
            if stage == 'preprocess':
                self.preprocess_R2_reads()

            elif stage == 'align':
                if self.use_memoized_outcomes:
                    self.make_uncommon_sequence_fastq()
                    read_type = 'collapsed_uncommon_R2'
                else:
                    read_type = 'collapsed_R2'

                self.generate_alignments(read_type)
                self.generate_supplemental_alignments_with_STAR(read_type, min_length=20)
                self.combine_alignments(read_type)

            elif stage == 'categorize':
                self.categorize_outcomes()
                self.collapse_UMI_outcomes()
                self.record_sanitized_category_names()
                #self.make_reads_per_UMI()
                #self.make_filtered_cell_bams()
                #self.extract_truncation_positions()
                self.extract_templated_insertion_info()
                self.extract_duplication_info()
                self.extract_genomic_insertion_info()
                #self.make_outcome_plots(num_examples=3)
            else:
                raise ValueError(stage)
        except:
            print(self.group, self.sample_name)
            raise

class SingleGuideNoUMIExperiment(SingleGuideExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.outcome_fn_keys = ['outcome_list']
    
    @property
    def final_Outcome(self):
        return coherence.gDNA_Outcome

    def merge_read_chunks(self):
        with gzip.open(self.fns_by_read_type['fastq']['collapsed_R2'], 'wt', compresslevel=1) as combined_fh:
            for read in self.reads:
                if read.Q30_fraction > 0.6:
                    combined_fh.write(str(read))

    @memoized_property
    def filtered_cell_outcomes(self):
        df = pd.read_csv(self.fns['outcome_list'], header=None, na_filter=False, names=coherence.gDNA_Outcome.columns, sep='\t')
        return df

    def collapse_UMI_outcomes(self):
        # poorly named for now
        counts = self.filtered_cell_outcomes.groupby(by=['category', 'subcategory', 'details']).size().sort_values(ascending=False)
        counts = pd.concat([counts], keys=[True], names=['perfect_guide'])
        counts.to_csv(self.fns['outcome_counts'], sep='\t', header=False)

        if self.fns['filtered_cell_outcomes'].exists():
            self.fns['filtered_cell_outcomes'].unlink()

        self.fns['filtered_cell_outcomes'].symlink_to(self.fns['outcome_list'])

    def extract_genomic_insertion_info(self):
        genomic_insertion_seqs = fasta.to_dict(self.fns['genomic_insertion_seqs'])

        with open(self.fns['outcome_list']) as outcomes_fh, \
             open(self.fns['filtered_genomic_insertion_seqs'], 'w') as seqs_fh, \
             open(self.fns['filtered_genomic_insertion_details'], 'w') as details_fh:

            for line in outcomes_fh:
                outcome = coherence.gDNA_Outcome.from_line(line)
                if outcome.category == 'genomic insertion' and outcome.subcategory == 'hg19':
                    name = outcome.standardized_qname
                    record = fasta.Record(name, genomic_insertion_seqs[name])
                    seqs_fh.write(str(record))

                    details_fh.write(f'{outcome.details}\n')

class CommonSequenceExperiment(SingleGuideExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_memoized_outcomes = False
    
    @property
    def results_dir(self):
        return self.pool.fns['common_sequences_dir'] / self.sample_name
    
    @memoized_property
    def target_name(self):
        prefix = self.description['target_info_prefix']
        target_name = prefix
        return target_name

    def process(self):
        try:
            read_type = 'collapsed_R2'
            self.generate_alignments(read_type)
            self.generate_supplemental_alignments_with_STAR(read_type, min_length=20)
            self.combine_alignments(read_type)
            self.categorize_outcomes()
        except:
            print(self.name)
            raise

    @memoized_property
    def outcomes(self):
        return coherence.load_UMI_outcomes(self.fns['outcome_list'], pooled=True)

def collapse_categories(df):
    possibly_collapse = ['genomic insertion', 'donor misintegration', 'complex duplication']
    to_collapse = [cat for cat in possibly_collapse if cat in df.index.levels[0]]

    new_rows = {}
    
    for category in to_collapse:
        subcats = sorted({s for c, s, v in df.index.values if c == category})
        for subcat in subcats:
            to_add = df.loc[category, subcat]
            new_rows[category, subcat, 'collapsed'] = to_add.sum()

    if any(c == 'donor' for c, s, d in df.index.values):
        all_details = set(d for s, d in df.loc['donor'].index.values)

        for details in all_details:
            new_rows['donor', 'collapsed', details] = df.loc['donor', :, details].sum()

        to_collapse.append('donor')

    df = df.drop(to_collapse, level=0)
    new_rows = pd.DataFrame.from_dict(new_rows, orient='index')

    return pd.concat((df, new_rows))

class PooledScreen:
    def __init__(self, base_dir, group, progress=None):
        self.base_dir = Path(base_dir)
        self.group = group

        if progress is None:
            def ignore_kwargs(x, **kwargs):
                return x
            progress = ignore_kwargs

        def pass_along_kwargs(iterable, **kwargs):
            return progress(iterable, **kwargs)

        self.progress = pass_along_kwargs

        sample_sheet_fn = self.base_dir / 'results' / group / 'sample_sheet.yaml'
        self.sample_sheet = yaml.safe_load(sample_sheet_fn.read_text())

        categorizer_name = self.sample_sheet.get('categorizer', 'pooled_layout')
        if categorizer_name == 'pooled_layout':
            self.categorizer = pooled_layout.Layout
        elif categorizer_name == 'prime_editing_layout': 
            self.categorizer = prime_editing_layout.Layout
        else:
            raise ValueError(categorizer_name)

        self.has_UMIs = self.sample_sheet.get('has_UMIs', True)

        self.short_name = self.sample_sheet.get('short_name', self.group)

        self.sgRNA = self.sample_sheet.get('sgRNA')
        self.donor = self.sample_sheet.get('donor')
        self.primer_names = self.sample_sheet.get('primer_names')
        self.sequencing_start_feature_name = self.sample_sheet.get('sequencing_start_feature_name', None)
        self.target_name = self.sample_sheet['target_info_prefix']

        self.supplemental_index_names = ['hg19', 'bosTau7']
        supplemental_indices = target_info.locate_supplemental_indices(self.base_dir)
        self.supplemental_indices = {name: supplemental_indices[name] for name in self.supplemental_index_names}

        self.min_reads_per_UMI = self.sample_sheet.get('min_reads_per_UMI', 4)

        self.target_info = target_info.TargetInfo(self.base_dir,
                                                  self.target_name,
                                                  donor=self.donor,
                                                  sgRNA=self.sgRNA,
                                                  primer_names=self.primer_names,
                                                  sequencing_start_feature_name=self.sequencing_start_feature_name,
                                                  supplemental_indices=self.supplemental_indices,
                                                  infer_homology_arms=self.sample_sheet.get('infer_homology_arms', False),
                                                 )

        self.variable_guide_library = guide_library.GuideLibrary(self.base_dir, self.sample_sheet['variable_guide_library'])
        self.variable_guides = self.variable_guide_library.guides

        if 'fixed_guide_library' in self.sample_sheet:
            self.fixed_guide_library = guide_library.GuideLibrary(self.base_dir, self.sample_sheet['fixed_guide_library'])
        else:
            self.fixed_guide_library = guide_library.dummy_guide_library

        self.fixed_guides = self.fixed_guide_library.guides

        self.group_dir = self.base_dir / 'results' / group

        self.fns = {
            'read_counts': self.group_dir / 'read_counts.txt',

            'outcome_counts': self.group_dir  / 'outcome_counts.npz',
            'total_outcome_counts': self.group_dir / 'total_outcome_counts.txt',
            'collapsed_outcome_counts': self.group_dir / 'collapsed_outcome_counts.npz',
            'collapsed_total_outcome_counts': self.group_dir / 'collapsed_total_outcome_counts.txt',

            'high_frequency_outcome_counts': self.group_dir / 'high_frequency_outcome_counts.hdf5',

            'filtered_cell_bam': self.group_dir / 'filtered_cell_alignments.bam',
            'reads_per_UMI': self.group_dir / 'reads_per_UMI.pkl',

            'quantiles': self.group_dir / 'quantiles.hdf5',

            'common_sequences_dir': self.group_dir / 'common_sequences',
            'common_name_to_common_sequence': self.group_dir / 'common_sequences' / 'common_name_to_common_sequence.txt',
            'all_sequences': self.group_dir / 'common_sequences' / f'{group}_all_sequences.txt',
            'common_sequence_outcomes': self.group_dir / 'common_sequences' / 'common_sequence_to_outcome.txt',
            'common_name_to_outcome': self.group_dir / 'common_sequences' / 'common_name_to_outcome.txt',

            'fraction_removed': self.group_dir / 'fraction_removed.hdf5',

            'common_sequence_special_alignments': self.group_dir / 'common_sequences' / 'all_special_alignments.bam',
            'special_alignments_dir': self.group_dir / 'special_alignments',

            'filtered_templated_insertion_details': self.group_dir / 'filtered_templated_insertion_details.hdf5',
            'filtered_duplication_details': self.group_dir / 'filtered_duplication_details.hdf5',

            'genomic_insertion_length_distributions': self.group_dir / 'genomic_insertion_length_distribution.txt',
        }

        label_offsets = {}
        for (seq_name, feature_name), feature in self.target_info.features.items():
            if feature.feature.startswith('donor_insertion'):
                label_offsets[feature_name] = 2
            elif feature.feature.startswith('donor_deletion'):
                label_offsets[feature_name] = 3
            elif feature_name.startswith('HA_RT'):
                label_offsets[feature_name] = 1

        label_offsets[f'{self.target_info.primary_sgRNA}_PAM'] = 2

        self.target_info.features_to_show.update(set(self.target_info.PAM_features))

        self.diagram_kwargs = dict(features_to_show=self.target_info.features_to_show,
                                   ref_centric=True,
                                   draw_sequence=True,
                                   flip_target=self.target_info.sequencing_direction == '-',
                                   #flip_donor=self.target_info.sgRNA_feature.strand == self.target_info.sequencing_direction,
                                   highlight_SNPs=True,
                                   split_at_indels=True,
                                   force_left_aligned=False,
                                   label_offsets=label_offsets,
                                  )

    @memoized_property
    def guide_combinations(self):
        combinations = []

        for fixed_guide in self.fixed_guides:
            for variable_guide in self.variable_guides:
                combinations.append((fixed_guide, variable_guide))

        return combinations

    @memoized_property
    def guide_combinations_by_read_count(self):
        read_counts = pd.read_csv(self.fns['read_counts'], sep='\t', index_col=[0, 1], squeeze=True)
        return [(fg, vg) for fg, vg in read_counts.index.values if fg != 'unknown' and vg != 'unknown']

    def guide_combinations_for_gene(self, gene):
        if isinstance(gene, (list, tuple)):
            return self.guide_combinations_for_gene_pair(gene)

        combinations = set()

        for fixed_guide in self.fixed_guide_library.gene_guides(gene):
            for variable_guide in self.variable_guide_library.gene_guides([gene, 'negative_control']):
                combinations.add((fixed_guide, variable_guide))

        for fixed_guide in self.fixed_guide_library.gene_guides('negative_control'):
            for variable_guide in self.variable_guide_library.gene_guides(gene):
                combinations.add((fixed_guide, variable_guide))

        return sorted(combinations)

    def guide_combinations_for_gene_pair(self, genes):
        combinations = set()

        first_gene, second_gene = genes

        for fixed_guide in self.fixed_guide_library.gene_guides(first_gene):
            for variable_guide in self.variable_guide_library.gene_guides(second_gene):
                combinations.add((fixed_guide, variable_guide))

        for fixed_guide in self.fixed_guide_library.gene_guides(second_gene):
            for variable_guide in self.variable_guide_library.gene_guides(first_gene):
                combinations.add((fixed_guide, variable_guide))

        return sorted(combinations)

    def guide_combinations_for_ordered_gene_pair(self, fixed_gene, variable_gene):
        combinations = []

        for fixed_guide in self.fixed_guide_library.gene_guides(fixed_gene):
            for variable_guide in self.variable_guide_library.gene_guides(variable_gene):
                combinations.append((fixed_guide, variable_guide))

        return sorted(combinations)

    def guide_plus_non_targeting(self, guide):
        combos = []

        if guide in self.fixed_guides:
            for variable_guide in self.variable_guide_library.non_targeting_guides:
                combos.append((guide, variable_guide))
        else:
            for fixed_guide in self.fixed_guide_library.non_targeting_guides:
                combos.append((fixed_guide, guide))

        return combos

    def single_guide_experiments(self, no_progress=False):
        for fixed_guide, variable_guide in self.guide_combinations:
            yield self.single_guide_experiment(fixed_guide, variable_guide, no_progress=no_progress)

    def single_guide_experiment(self, fixed_guide, variable_guide, no_progress=False):
        if no_progress:
            progress = None
        else:
            progress = self.progress

        return SingleGuideExperiment(self.base_dir, self.group, fixed_guide, variable_guide, pool=self, progress=progress)

    @memoized_property
    def R2_read_length(self):
        exp = next(self.single_guide_experiments(no_progress=True))
        return exp.R2_read_length

    @memoized_property
    def blunt_insertion_length_detection_limit(self):
        return self.R2_read_length - (self.target_info.sequencing_start.end - self.target_info.cut_after) - 5

    def make_outcome_counts(self):
        all_counts = {}

        description = 'Loading outcome counts'
        guide_combos = self.progress(self.guide_combinations, desc=description)
        for fixed_guide, variable_guide in guide_combos:
            exp = SingleGuideExperiment(self.base_dir, self.group, fixed_guide, variable_guide, pool=self)
            try:
                all_counts[fixed_guide, variable_guide] = exp.outcome_counts
            except (FileNotFoundError, pd.errors.EmptyDataError):
                pass

        all_outcomes = set()

        for fixed_guide, variable_guide in all_counts:
            all_outcomes.update(all_counts[fixed_guide, variable_guide].index.values)
            
        outcome_order = sorted(all_outcomes)
        outcome_to_index = {outcome: i for i, outcome in enumerate(outcome_order)}

        counts = scipy.sparse.dok_matrix((len(outcome_order), len(self.guide_combinations)), dtype=int)

        description = 'Combining outcome counts'
        guide_combos = self.progress(self.guide_combinations, desc=description)
        for g, (fixed_guide, variable_guide) in enumerate(guide_combos):
            if (fixed_guide, variable_guide) in all_counts:
                for outcome, count in all_counts[fixed_guide, variable_guide].items():
                    o = outcome_to_index[outcome]
                    counts[o, g] = count
                
        scipy.sparse.save_npz(self.fns['outcome_counts'], counts.tocoo())

        df = pd.DataFrame(counts.todense(),
                          columns=self.guide_combinations,
                          index=pd.MultiIndex.from_tuples(outcome_order),
                         )

        df.sum(axis=1).to_csv(self.fns['total_outcome_counts'], header=False)

        # Collapse potentially equivalent outcomes together.
        collapsed = pd.concat({pg: collapse_categories(df.loc[pg]) for pg in [True, False] if pg in df.index.levels[0]})

        coo = scipy.sparse.coo_matrix(np.array(collapsed))
        scipy.sparse.save_npz(self.fns['collapsed_outcome_counts'], coo)

        collapsed.sum(axis=1).to_csv(self.fns['collapsed_total_outcome_counts'], header=False)

    @memoized_with_key
    def total_outcome_counts(self, collapsed):
        if collapsed:
            prefix = 'collapsed_'
        else:
            prefix = ''

        key = prefix + 'total_outcome_counts'

        return pd.read_csv(self.fns[key], header=None, index_col=[0, 1, 2, 3], na_filter=False)

    @memoized_with_key
    def outcome_counts_df(self, collapsed):
        guides = self.guide_combinations

        if collapsed:
            prefix = 'collapsed_'
        else:
            prefix = ''

        key = prefix + 'outcome_counts'

        sparse_counts = scipy.sparse.load_npz(self.fns[key])
        df = pd.DataFrame(sparse_counts.todense(),
                          index=self.total_outcome_counts(collapsed).index,
                          columns=pd.MultiIndex.from_tuples(guides),
                         )

        df.index.names = ('perfect_guide', 'category', 'subcategory', 'details')
        df.columns.names = ('fixed_guide', 'variable_guide')

        return df

    @memoized_with_key
    def outcome_counts(self, guide_status):
        if guide_status == 'all':
            outcome_counts = self.outcome_counts_df(True).sum(level=[1, 2, 3])
        elif guide_status == 'perfect':
            outcome_counts = self.outcome_counts_df(True).loc[True]
        else:
            perfect_guide = guide_status == 'perfect'
            outcome_counts = self.outcome_counts_df(True).loc[perfect_guide]

        return outcome_counts

    def make_high_frequency_outcome_counts(self):
        outcomes = self.most_frequent_outcomes('none')

        non_targeting_fractions = self.non_targeting_fractions('perfect', 'none').loc[outcomes]

        to_write = {
            'counts': self.outcome_counts('perfect').loc[outcomes],
            'fractions': self.outcome_fractions('perfect').loc[outcomes],
            'log2_fold_changes': self.log2_fold_changes('perfect', 'none').loc[outcomes],
        }

        fractions_intervals = {
            'lower': [],
            'upper': [],
        } 

        UMI_counts = self.UMI_counts('perfect')

        for _, row in to_write['counts'].iterrows():
            lowers, uppers = utilities.clopper_pearson_fast(row, UMI_counts)
            fractions_intervals['lower'].append(lowers)
            fractions_intervals['upper'].append(uppers)

        for key, array in fractions_intervals.items():
            fractions = pd.DataFrame(array, index=to_write['counts'].index)

            fold_changes = fractions.div(non_targeting_fractions, axis=0)
            fold_changes = fold_changes.fillna(2**5).replace(0, 2**-5)
            log2_fold_changes = np.log2(fold_changes)

            to_write.update({
                f'fractions_interval_{key}': fractions,
                f'log2_fold_changes_interval_{key}': log2_fold_changes,
            })

        with h5py.File(self.fns['high_frequency_outcome_counts'], 'w') as fh:
            for key, df in to_write.items():
                dataset = fh.create_dataset(key, data=df.values)
                
                for level in df.index.names:
                    dataset.attrs[level] = [s.encode() for s in df.index.get_level_values(level)]
                    
                for level in df.columns.names:
                    dataset.attrs[level] = [s.encode() for s in df.columns.get_level_values(level)]

    def load_high_frequency_outcome_counts(self, key):
        with h5py.File(self.fns['high_frequency_outcome_counts']) as fh:
            index_level_names = ['category', 'subcategory', 'details']
            columns_level_names = ['fixed_guide', 'variable_guide']
            
            dataset = fh[key]
            index_levels = [[b.decode() for b in dataset.attrs[level]] for level in index_level_names]
            index = pd.MultiIndex.from_tuples(zip(*index_levels))
            
            columns_levels = [[b.decode() for b in dataset.attrs[level]] for level in columns_level_names]
            columns = pd.MultiIndex.from_tuples(zip(*columns_levels))
            
            outcome_counts = pd.DataFrame(dataset.value, index=index, columns=columns)
            
        return outcome_counts

    @memoized_property
    def high_frequency_outcome_counts(self):
        return self.load_high_frequency_outcome_counts('counts')

    @memoized_property
    def high_frequency_outcome_fractions(self):
        return self.load_high_frequency_outcome_counts('fractions')

    @memoized_property
    def high_frequency_log2_fold_changes(self):
        return self.load_high_frequency_outcome_counts('log2_fold_changes')

    @memoized_property
    def high_frequency_log2_fold_change_intervals(self):
        intervals = {
            k: self.load_high_frequency_outcome_counts(f'log2_fold_changes_interval_{k}')
            for k in ('lower', 'upper')
        }
        return pd.concat(intervals, axis=1)

    def compute_fraction_removed(self):
        ti = self.target_info
        deletion_fractions = self.outcome_fractions('perfect').xs('deletion', drop_level=False)

        outcome_to_deletion_slice = {}

        for c, s, d in deletion_fractions.index.values:
            # Undo anchor shift to make coordinates relative to full target sequence.
            deletion = pooled_layout.DeletionOutcome.from_string(d).perform_anchor_shift(-ti.anchor).deletion
            start = min(deletion.starts_ats)
            stop = max(deletion.ends_ats) + 1
            deletion_slice = slice(start, stop)
            outcome_to_deletion_slice[c, s, d] = deletion_slice

        index = np.arange(len(ti.target_sequence))
        columns = deletion_fractions.columns

        fraction_removed = np.zeros((len(index), len(columns)))

        for outcome, deletion_slice in self.progress(outcome_to_deletion_slice.items()):
            fraction_removed[deletion_slice] += deletion_fractions.loc[outcome]  

        fraction_removed = pd.DataFrame(fraction_removed, index=index, columns=columns)

        left_edge = ti.features[ti.target, 'polyT-track'].end
        right_edge = ti.features[ti.target, 'sequencing_start'].start

        relevant_window = fraction_removed.loc[left_edge:right_edge]
        relevant_window.index = relevant_window.index - ti.cut_after

        relevant_window.to_hdf(self.fns['fraction_removed'], 'fraction_removed')

    @memoized_property
    def fraction_removed(self):
        return pd.read_hdf(self.fns['fraction_removed'])

    @memoized_property
    def fraction_removed_log2_fold_changes(self):
        fraction_removed = self.fraction_removed

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='divide by zero encountered in log2', category=RuntimeWarning)
            log2_fold_changes = np.log2(fraction_removed.div(fraction_removed['all_non_targeting', 'all_non_targeting'], axis=0))

        return log2_fold_changes
            
    @memoized_property
    def non_targeting_guide_pairs(self):
        pairs = []
        for fixed_nt in self.fixed_guide_library.non_targeting_guides:
            for variable_nt in self.variable_guide_library.non_targeting_guides:
                pairs.append((fixed_nt, variable_nt))
        return pairs

    @memoized_with_key
    def UMI_counts(self, guide_status):
        return self.outcome_counts(guide_status).sum()
    
    @memoized_with_key
    def outcome_fractions(self, guide_status):
        per_guide_fractions = self.outcome_counts(guide_status) / self.UMI_counts(guide_status)
        
        all_nt_fractions = [self.non_targeting_fractions(guide_status, fixed_guide) for fixed_guide in list(self.fixed_guides) + [ALL_NON_TARGETING]]
        return pd.concat([per_guide_fractions] + all_nt_fractions, axis=1)
    
    @memoized_with_key
    def non_targeting_outcomes(self, fixed_guide):
        guide_outcomes = {}
        for nt_guide in self.variable_guide_library.non_targeting_guides:
            exp = SingleGuideExperiment(self.base_dir, self.group, fixed_guide, nt_guide)
            fn = exp.fns['filtered_cell_outcomes']

            outcomes = [coherence.Pooled_UMI_Outcome.from_line(line) for line in fn.open()]

            for outcome in outcomes:
                if outcome.category == 'genomic insertion':
                    outcome.details = 'n/a'
                
                if outcome.category == 'donor insertion':
                    outcome.details = 'n/a'

            guide_outcomes[nt_guide] = outcomes

        return guide_outcomes

    @memoized_with_key
    def non_targeting_counts(self, guide_status, fixed_guide):
        if fixed_guide is None:
            fixed_guide = ALL_NON_TARGETING

        if fixed_guide is ALL_NON_TARGETING:
            fixed_nts = self.fixed_guide_library.non_targeting_guides
        else:
            fixed_nts = fixed_guide
        
        variable_nts = self.variable_guide_library.non_targeting_guides

        counts = self.outcome_counts(guide_status).loc(axis=1)[fixed_nts, variable_nts]
        return counts.sum(axis='columns').sort_values(ascending=False)

    @memoized_with_key
    def non_targeting_fractions(self, guide_status, fixed_guide):
        counts = self.non_targeting_counts(guide_status, fixed_guide)
        fractions = counts / counts.sum()
        fractions.name = (fixed_guide, ALL_NON_TARGETING)
        return fractions

    @memoized_with_key
    def most_frequent_outcomes(self, fixed_guide):
        return self.non_targeting_counts('all', fixed_guide).index.values[:1000]

    @memoized_with_key
    def common_counts(self, guide_status):
        # Regardless of guide_status, use 'all' to define common non-targeting outcomes.
        common_counts = self.outcome_counts(guide_status).loc[self.most_frequent_outcomes] 
        leftover = self.UMI_counts(guide_status) - common_counts.sum()
        leftover_row = pd.DataFrame.from_dict({('uncommon', 'uncommon', 'collapsed'): leftover}, orient='index')
        common_counts = pd.concat([common_counts, leftover_row])
        return common_counts
    
    @memoized_property
    def common_non_targeting_counts(self):
        return self.common_counts('perfect')[self.variable_guide_library.non_targeting_guides].sum(axis=1)
    
    @memoized_property
    def common_non_targeting_fractions(self):
        counts = self.common_non_targeting_counts
        return counts / counts.sum()
    
    @memoized_with_key
    def common_fractions(self, guide_status):
        return self.common_counts(guide_status) / self.UMI_counts(guide_status)

    @memoized_with_key
    def fold_changes(self, guide_status, fixed_guide):
        if fixed_guide is None:
            fixed_guide = ALL_NON_TARGETING
        fractions = self.outcome_fractions(guide_status)
        denominator = fractions[fixed_guide, ALL_NON_TARGETING]
        return fractions.div(denominator, axis=0)

    @memoized_with_key
    def log2_fold_changes(self, guide_status, fixed_guide):
        fc = self.fold_changes(guide_status, fixed_guide)
        fc = fc.fillna(2**5).replace(0, 2**-5)
        return np.log2(fc)
    
    def log2_fold_changes_multiple_outcomes(self, outcomes, fixed_guide='none', guide_status='perfect'):
        fractions = self.outcome_fractions(guide_status)[fixed_guide].loc[outcomes].sum(axis=0)
        nt_fraction = self.non_targeting_fractions(guide_status, fixed_guide).loc[outcomes].sum()
        fc = fractions / nt_fraction
        fc = fc.fillna(2**5).replace(0, 2**-5)
        return np.log2(fc)

    def donor_outcomes_containing_SNV(self, SNV_name):
        ti = self.target_info
        SNV_index = sorted(ti.donor_SNVs['target']).index(SNV_name)
        donor_base = ti.donor_SNVs['donor'][SNV_name]['base']
        outcomes = [(c, s, d) for c, s, d in self.outcome_counts('perfect').index if c == 'donor' and d[SNV_index] == donor_base]
        return outcomes

    @memoized_property
    def conversion_fractions(self):
        conversion_fractions = {}

        SNVs = self.target_info.donor_SNVs['target']

        outcome_fractions = self.outcome_fractions('perfect')['none']

        for SNV_name in SNVs:
            outcomes = self.donor_outcomes_containing_SNV(SNV_name)
            fractions = outcome_fractions.loc[outcomes].sum()
            conversion_fractions[SNV_name] = fractions

        conversion_fractions = pd.DataFrame.from_dict(conversion_fractions, orient='index').sort_index()
        
        return conversion_fractions

    @memoized_property
    def conversion_log2_fold_changes(self):
        fractions = self.conversion_fractions
        fold_changes = fractions.div(fractions['all_non_targeting'], axis='index').drop(columns=['all_non_targeting'])
        log2_fold_changes = np.log2(fold_changes)
        return log2_fold_changes

    @memoized_property
    def SNV_name_to_position(self):
        SNVs = self.target_info.donor_SNVs['target']
        
        SNV_name_to_position = {}
        for SNV_name, SNV in SNVs.items():
            position = SNVs[SNV_name]['position']
            SNV_name_to_position[SNV_name] = position
            
        return pd.Series(SNV_name_to_position).sort_index()
        
    def rational_outcome_order(self, fixed_guide, num_outcomes=50, include_uncommon=False, by_frequency=False):
        def get_deletion_info(details):
            deletion = target_info.degenerate_indel_from_string(details)
            return {'num_MH_nts': len(deletion.starts_ats) - 1,
                    'start': min(deletion.starts_ats),
                    'length': deletion.length,
                    }

        def has_MH(details):
            info = get_deletion_info(details)
            return info['num_MH_nts'] >= 2 and info['length'] > 1

        conditions = {
            'insertions': lambda c, sc, d: c == 'insertion',
            'no_MH_deletions': lambda c, sc, d: c == 'deletion' and not has_MH(d),
            'MH_deletions': lambda c, sc, d: c == 'deletion' and has_MH(d),
            'donor': lambda c, sc, d: c == 'donor' and sc == 'collapsed',
            'wt': lambda c, sc, d: c == 'wild type' and sc != 'mismatches' and d != '____----',
            'uncat': lambda c, sc, d: c == 'uncategorized',
            'genomic': lambda c, sc, d: c == 'genomic insertion',
            'donor misintegration': lambda c, sc, d: c == 'donor misintegration',
            'complex templated insertion': lambda c, sc, d: c == 'complex templated insertion',
            'SD-MMEJ': lambda c, sc, d: c == 'SD-MMEJ',
            'uncommon': [('uncommon', 'uncommon', 'collapsed')],
        }

        group_order = [
            'wt',
            'donor',
            'insertions',
            'no_MH_deletions',
            'MH_deletions',
            'SD-MMEJ',
            'uncat',
            'genomic',
            'donor misintegration',
            'complex templated insertion',
        ]
        if include_uncommon:
            group_order.append('uncommon')

        donor_order = [
            'ACGAGTTT',
            '___AGTTT',
            '____GTTT',
            '___AGTT_',
            '____GTT_',
            '____GT__',
            '____G___',
            'ACGAGTT_',
            'ACGAGT__',
            'ACGAG___',
            'ACGA____',
            'ACG_____',
            'ACG_GTTT',
            'ACAAGTTT',
            'ACG',
            '___',
        ]

        groups = {
            name: [o for o in self.most_frequent_outcomes(fixed_guide)[:num_outcomes] if condition(*o)] if name != 'uncommon' else condition
            for name, condition in conditions.items()
        }

        def donor_key(csd):
            details = csd[2]
            if ';' in details:
                variable_locii_details, deletion_details = details.split(';', 1)
            else:
                variable_locii_details = details
                deletion_details = None

            if variable_locii_details in donor_order:
                i = donor_order.index(variable_locii_details)
            else:
                i = 1000
            return i, deletion_details

        def deletion_key(csd):
            details = csd[2]
            length = get_deletion_info(details)['length']
            return length

        if not by_frequency:
            groups['donor'] = sorted(groups['donor'], key=donor_key)
            for k in ['no_MH_deletions', 'MH_deletions']:
                groups[k] = sorted(groups[k], key=deletion_key)

        ordered = []
        for name in group_order:
            ordered.extend(groups[name])

        sizes = [len(groups[name]) for name in group_order]
        return ordered, sizes

    def merge_filtered_bams(self):
        input_fns = []
        for guide in self.variable_guide_library.non_targeting_guides:
            exp = SingleGuideExperiment(self.base_dir, self.group, guide)
            input_fns.append(exp.fns['filtered_cell_bam'])

        sam.merge_sorted_bam_files(input_fns, self.fns['filtered_cell_bam'])
    
    def merge_common_sequence_special_alignments(self):
        chunks = self.common_sequence_chunks()

        all_fns = []

        for chunk in chunks:
            for sub_dir in chunk.fns['outcomes_dir'].iterdir():
                fn = sub_dir / 'special_alignments.bam'
                if fn.exists():
                    all_fns.append(fn)

        if len(all_fns) > 0:
            sam.merge_sorted_bam_files(all_fns, self.fns['common_sequence_special_alignments'])

    def merge_special_alignments(self):
        all_fns = defaultdict(dict)

        for exp in self.progress(self.single_guide_experiments(), desc='Finding files'):
            for sub_dir in exp.fns['outcomes_dir'].iterdir():
                outcome = sub_dir.name
                fn = sub_dir / 'special_alignments.bam'
                if fn.exists():
                    all_fns[outcome][exp.name] = fn

        top_dir = self.fns['special_alignments_dir']

        top_dir.mkdir(exist_ok=True)

        description = 'Merging special alignments'
        for outcome, outcome_fns in self.progress(all_fns.items(), desc=description):
            first_fn = list(outcome_fns.values())[0]
            with pysam.AlignmentFile(first_fn) as fh:
                header = fh.header
                
            merged_bam_fn = top_dir / f'{outcome}.bam'
            with sam.AlignmentSorter(merged_bam_fn, header) as sorter:
                for guide, fn in outcome_fns.items():
                    with pysam.AlignmentFile(fn) as individual_fh:
                        for al in individual_fh:
                            al.query_name = f'{al.query_name}_{guide}'
                            sorter.write(al)

    def merge_reads_per_UMI(self):
        reads_per_UMI = defaultdict(Counter)

        description = 'Merging reads per UMI'
        total = len(self.guide_combinations)
        for exp in self.progress(self.single_guide_experiments(), desc=description, total=total):
            for category, counts in exp.reads_per_UMI.items():
                reads_per_UMI[category].update(counts)

        with open(str(self.fns['reads_per_UMI']), 'wb') as fh:
            pickle.dump(dict(reads_per_UMI), fh)

    def merge_templated_insertion_details(self, fn_key='filtered_templated_insertion_details'):
        with h5py.File(self.fns[fn_key], 'w') as merged_f:
            description = 'Merging templated insertion details'
            total = len(self.guide_combinations)

            all_counts = defaultdict(Counter)

            for exp in self.progress(self.single_guide_experiments(), desc=description, total=total):

                def add_to_merged(key, dataset):
                    fields = key.split('/')

                    if len(fields) == 3:
                        all_counts[key].update(dict(zip(dataset['values'], dataset['counts'])))

                    if len(fields) == 4:
                        category, subcategory, field, array_type = fields

                        if array_type == 'list':
                            return

                        new_key = f'{category}/{subcategory}/{field}/{exp.name}/{array_type}'

                        merged_f.create_dataset(new_key, data=dataset.value)

                with h5py.File(exp.fns[fn_key]) as exp_f:
                    exp_f.visititems(add_to_merged)

            for key, counts in all_counts.items():
                
                new_key = f'{key}/all'

                values = np.array(sorted(counts))
                counts = np.array([counts[v] for v in values])

                merged_f.create_dataset(f'{new_key}/values', data=values)
                merged_f.create_dataset(f'{new_key}/counts', data=counts)

    def extract_genomic_insertion_length_distributions(self):
        all_guide_combos = {(fg, vg): (fg, vg) for fg, vg in self.guide_combinations}
        all_guide_combos[(ALL_NON_TARGETING, ALL_NON_TARGETING)] = self.non_targeting_guide_pairs

        for fg in self.fixed_guides:
            all_guide_combos[(fg, ALL_NON_TARGETING)] = self.guide_plus_non_targeting(fg)

        length_distributions = {}

        read_length = self.R2_read_length
        
        for guide_name, guide_combos in self.progress(all_guide_combos.items()):    
            for organism in ['hg19', 'bosTau7']:
                lengths, counts = self.templated_insertion_details(guide_combos, 'genomic insertion', organism, 'insertion_length')
            
                if len(lengths) == 0:
                    continue
                    
                lengths_array = np.zeros(read_length + 1)
            
                for l, c in zip(lengths, counts):
                    lengths_array[l] = c
            
                UMIs = self.UMI_counts('perfect').loc[guide_combos].sum()
            
                key = tuple([organism] + list(guide_name))
                length_distributions[key] = lengths_array / UMIs

        length_distributions_df = pd.DataFrame(length_distributions).T

        length_distributions_df.to_csv(self.fns['genomic_insertion_length_distributions'])

    @memoized_property
    def genomic_insertion_length_distributions(self):
        df = pd.read_csv(self.fns['genomic_insertion_length_distributions'], index_col=[0, 1, 2])
        df.columns = [int(c) for c in df.columns]
        df.index.names = ['organism', 'fixed_guide', 'variable_guide']
        return df
        
    def templated_insertion_details(self, guide_pairs, category, subcategories, field, fn_key='filtered_templated_insertion_details'):
        counts = Counter()

        if isinstance(subcategories, str):
            subcategories = [subcategories]

        if guide_pairs == 'all':
            guide_pair_keys = ['all']
        else:
            if isinstance(guide_pairs, tuple):
                guide_pairs = [guide_pairs]

            guide_pair_keys = [f'{fixed_guide}-{variable_guide}' for fixed_guide, variable_guide in guide_pairs]

        with h5py.File(self.fns[fn_key]) as f:
            for guide_pair_key in guide_pair_keys:
                for subcategory in subcategories:
                    group = f'{category}/{subcategory}/{field}/{guide_pair_key}'
                    if group in f:
                        counts.update(dict(zip(f[group]['values'], f[group]['counts'])))

        if pooled_layout.NAN_INT in counts:
            counts.pop(pooled_layout.NAN_INT)

        if len(counts) == 0:
            xs = np.array([])
        else:
            xs = np.arange(min(counts), max(counts) + 1)

        ys = np.array([counts[x] for x in xs])

        return xs, ys

    @memoized_property
    def reads_per_UMI(self):
        with open(str(self.fns['reads_per_UMI']), 'rb') as fh:
            reads_per_UMI = pickle.load(fh)

        for category, counts in reads_per_UMI.items():
            reads_per_UMI[category] = utilities.counts_to_array(counts)

        return reads_per_UMI

    def chi_squared_per_guide(self, relevant_outcomes=None, fixed_guide='none', exclude_unedited=True):
        if relevant_outcomes is None:
            relevant_outcomes = 50
        if isinstance(relevant_outcomes, int):
            relevant_outcomes = self.most_frequent_outcomes(fixed_guide)[:relevant_outcomes]
            
        if exclude_unedited:
            relevant_outcomes = [outcome for outcome in relevant_outcomes if outcome != ('wild type', 'clean', 'n/a')]

        counts = self.outcome_counts('perfect')[fixed_guide].loc[relevant_outcomes]
        
        # A column with zero counts causes problems.
        guide_counts = counts.sum()
        nonzero_guides = guide_counts[guide_counts > 0].index
        counts = counts[nonzero_guides]
        
        non_targeting_guides = sorted(set(self.variable_guide_library.non_targeting_guides) & set(nonzero_guides))
        
        UMI_counts = counts.sum()
        nt_totals = counts[non_targeting_guides].sum(axis=1)
        nt_fractions = nt_totals / nt_totals.sum()
        expected = pd.DataFrame(np.outer(nt_fractions, UMI_counts), index=counts.index, columns=nonzero_guides)
        difference = counts - expected
        return (difference**2 / expected).sum().sort_values(ascending=False)

    def explore(self, **kwargs):
        return explore(self.base_dir, self.group, **kwargs)

    def make_common_sequences(self):
        splitter = CommonSequenceSplitter(self)

        Annotation = collapse.Annotations['collapsed_UMI_mismatch']
        def Read_to_num_reads(r):
            return Annotation.from_identifier(r.name)['num_reads']

        description = 'Collecting common sequences'
        exps = self.single_guide_experiments(no_progress=True)
        for exp in self.progress(exps, desc=description, total=len(self.guide_combinations)):
            reads = exp.reads_by_type('collapsed_R2')
            enough_reads_per_UMI = (r.seq for r in reads if Read_to_num_reads(r) >= 5)
            splitter.update_counts(enough_reads_per_UMI)

        splitter.write_files()

    @memoized_property
    def common_sequence_chunk_names(self):
        def d_to_chunk_name(d):
            return d.name[len('common_sequences-'):]

        return sorted([d_to_chunk_name(d) for d in self.fns['common_sequences_dir'].iterdir() if d.is_dir()])

    def common_sequence_chunks(self):
        for chunk_name in self.common_sequence_chunk_names:
            yield CommonSequenceExperiment(self.base_dir, self.group, 'common_sequences', chunk_name, pool=self, progress=self.progress)

    @memoized_property
    def common_name_to_common_sequence(self):
        name_to_seq = {}
        with self.fns['common_name_to_common_sequence'].open() as fh:
            for line in fh:
                name, seq = line.strip().split()
                name_to_seq[name] = seq

        return name_to_seq
    
    @memoized_property
    def common_names(self):
        common_names = []

        with self.fns['common_name_to_common_sequence'].open() as fh:
            for line in fh:
                name, seq = line.strip().split()
                common_names.append(name)

        return common_names

    @memoized_property
    def common_name_to_outcome(self):
        name_to_outcome = {}
        with self.fns['common_name_to_outcome'].open() as fh:
            for line in fh:
                name, category, subcategory, details = line.strip().split('\t')
                name_to_outcome[name] = (category, subcategory, details)

        return name_to_outcome

    @memoized_property
    def outcome_to_common_names(self):
        outcome_to_common_names = defaultdict(list)

        for common_name, outcome in self.common_name_to_outcome.items():
            outcome_to_common_names[outcome].append(common_name)

        return outcome_to_common_names
    
    @memoized_property
    def common_sequence_to_common_name(self):
        return utilities.reverse_dictionary(self.common_name_to_common_sequence)

    @memoized_property
    def common_name_to_special_alignment(self):
        name_to_al = {}

        if self.fns['common_sequence_special_alignments'].exists():
            for al in pysam.AlignmentFile(self.fns['common_sequence_special_alignments']):
                name_to_al[al.query_name] = al

        return name_to_al

    @memoized_property
    def common_sequence_to_special_alignment(self):
        name_to_al = self.common_name_to_special_alignment
        seq_to_name = self.common_sequence_to_common_name
        return {seq: name_to_al[name] for seq, name in seq_to_name.items() if name in name_to_al}

    @memoized_property
    def name_to_chunk(self):
        names = self.common_sequence_chunk_names
        starts = [int(n.split('-')[0]) for n in names]
        chunks = [CommonSequenceExperiment(self.base_dir, self.group, 'common_sequences', n, pool=self) for n in names]
        Annotation = collapse.Annotations['collapsed_UMI_mismatch']

        def name_to_chunk(name):
            number = int(Annotation.from_identifier(name)['UMI'])
            start_index = bisect.bisect(starts, number) - 1 
            chunk = chunks[start_index]
            return chunk

        return name_to_chunk

    def get_read_alignments(self, name):
        if isinstance(name, int):
            name = self.common_names[name]

        chunk = self.name_to_chunk(name)

        als = chunk.get_read_alignments(name)

        return als

    def get_read_layout(self, name, **kwargs):
        als = self.get_read_alignments(name)
        l = self.categorizer(als, self.target_info, **kwargs)
        return l

    def get_read_diagram(self, read_id, only_relevant=True, **diagram_kwargs):
        layout = self.get_read_layout(read_id)

        if only_relevant:
            layout.categorize()
            to_plot = layout.relevant_alignments
        else:
            to_plot = layout.alignments
            
        for k, v in self.diagram_kwargs.items():
            diagram_kwargs.setdefault(k, v)

        diagram = visualize.ReadDiagram(to_plot, self.target_info, **diagram_kwargs)

        return diagram

    def get_common_seq_alignments(self, seq):
        name = self.common_sequence_to_common_name[seq]
        als = self.get_read_alignments(name)
        return als

    @memoized_property
    def common_sequence_outcomes(self):
        outcomes = []
        for exp in self.common_sequence_chunks():
            outcomes.extend(exp.outcomes)

        return outcomes

    def write_common_outcome_files(self):
        with self.fns['common_sequence_outcomes'].open('w') as seq_fh, \
             self.fns['common_name_to_outcome'].open('w') as name_fh:

            for outcome in self.common_sequence_outcomes:
                common_name = outcome.original_name
                common_seq = self.common_name_to_common_sequence[common_name]
                outcome_fields = [
                    outcome.category,
                    outcome.subcategory,
                    outcome.details,
                ]
                outcome = '\t'.join(outcome_fields)

                seq_fh.write('{}\t{}\n'.format(common_seq, outcome))
                name_fh.write('{}\t{}\n'.format(common_name, outcome))

    @memoized_property
    def common_sequence_to_outcome(self):
        common_sequence_to_outcome = {}

        with self.fns['common_sequence_outcomes'].open() as fh:
            for line in fh:
                seq, category, subcategory, details = line.strip().split('\t')
                common_sequence_to_outcome[seq] = (category, subcategory, details)

        return common_sequence_to_outcome

class PooledScreenNoUMI(PooledScreen):
    def single_guide_experiment(self, fixed_guide, variable_guide, no_progress=False):
        if no_progress:
            progress = None
        else:
            progress = self.progress

        return SingleGuideNoUMIExperiment(self.base_dir, self.group, fixed_guide, variable_guide, pool=self, progress=progress)

    def make_common_sequences(self):
        splitter = CommonSequenceSplitter(self)

        description = 'Collecting common sequences'
        exps = self.single_guide_experiments(no_progress=True)
        for exp in self.progress(exps, desc=description, total=len(self.guide_combinations)):
            reads = exp.reads_by_type('collapsed_R2')
            seqs = (r.seq for r in reads)
            splitter.update_counts(seqs)

        splitter.write_files()

class CommonSequenceSplitter:
    def __init__(self, pool, reads_per_chunk=1000):
        self.pool = pool
        self.reads_per_chunk = reads_per_chunk
        self.current_chunk_fh = None
        self.seq_counts = Counter()
        self.distinct_guides_per_seq = Counter()
        
        common_sequences_dir = self.pool.fns['common_sequences_dir']

        if common_sequences_dir.is_dir():
            old_dir_suffix = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
            common_sequences_dir.rename(f'{common_sequences_dir}_{old_dir_suffix}')
            
        common_sequences_dir.mkdir()

    def update_counts(self, seqs):
        counts = Counter(seqs)
        self.seq_counts.update(counts)
        for seq in counts:
            self.distinct_guides_per_seq[seq] += 1
            
    def close(self):
        if self.current_chunk_fh is not None:
            self.current_chunk_fh.close()
            
    def possibly_make_new_chunk(self, i):
        if i % self.reads_per_chunk == 0:
            self.close()
            chunk_name = f'{i:010d}-{i + self.reads_per_chunk - 1:010d}'
            chunk_exp = CommonSequenceExperiment(self.pool.base_dir, self.pool.group, 'common_sequences', chunk_name)
            fn = chunk_exp.fns_by_read_type['fastq']['collapsed_R2']
            self.current_chunk_fh = gzip.open(fn, 'wt', compresslevel=1)
            
    def write_read(self, i, read):
        self.possibly_make_new_chunk(i)
        self.current_chunk_fh.write(str(read))
        
    def write_files(self):
        seq_lengths = {len(s) for s in self.seq_counts}
        if len(seq_lengths) > 1:
            raise ValueError('More than one sequence length: ', seq_lengths)
        seq_length = seq_lengths.pop()

        # Include one value outside of the solexa range to allow automatic detection.
        qual = fastq.encode_sanger([25] + [40] * (seq_length - 1))
   
        tuples = []

        Annotation = collapse.Annotations['collapsed_UMI_mismatch']

        i = 0 
        for seq, count in self.seq_counts.most_common():
            distinct_guides = self.distinct_guides_per_seq[seq]

            if count > 1 and distinct_guides > 1:
                name = str(Annotation(UMI=f'{i:010}', cluster_id=0, num_reads=count, mismatch=-1))
                read = fastq.Read(name, seq, qual)
                self.write_read(i, read)
                i += 1
            else:
                name = None

            tuples.append((name, seq, count))

        self.close()

        with self.pool.fns['common_name_to_common_sequence'].open('w') as name_to_seq_fh, \
             self.pool.fns['all_sequences'].open('w') as all_sequences_fh:

            for name, seq, count in tuples:
                all_sequences_fh.write(f'{seq}\t{count}\n')

                if name is not None:
                    name_to_seq_fh.write(f'{name}\t{seq}\n')

def explore(base_dir, group,
            initial_guide=None,
            by_outcome=True,
            fixed_guide='none',
            clear_output=True,
            **kwargs):
    pool = get_pool(base_dir, group)

    guides = pool.variable_guide_library.guides
    if initial_guide is None:
        initial_guide = guides[0]

    Select = ipywidgets.Select
    Layout = ipywidgets.Layout

    widgets = {
        'guide': Select(options=guides, value=initial_guide, layout=Layout(height='200px', width='300px')),
        'read_id': Select(options=[], layout=Layout(height='200px', width='600px')),
        'outcome': Select(options=[], continuous_update=False, layout=Layout(height='200px', width='450px')),
    }
    
    non_widgets = {
        'file_name': ipywidgets.Text(value=str(Path(base_dir) / 'figures')),
        'save': ipywidgets.Button(description='Save snapshot'),
    }

    toggles = [
        ('parsimonious', False),
        ('relevant', True),
        ('ref_centric', True),
        ('draw_sequence', False),
        ('draw_qualities', False),
        ('draw_mismatches', True),
        ('draw_read_pair', False),
        ('force_left_aligned', False),
        ('split_at_indels', False),
        ('highlight_SNPs', True),
    ]
    for key, default_value in toggles:
        widgets[key] = ipywidgets.ToggleButton(value=kwargs.pop(key, default_value))

    # For some reason, the target widget doesn't get a label without this.
    for k, v in widgets.items():
        v.description = k

    output = ipywidgets.Output()

    def get_exp():
        guide = widgets['guide'].value
        exp = pool.single_guide_experiment(fixed_guide, guide)
        return exp

    @output.capture()
    def populate_outcomes(change):
        previous_value = widgets['outcome'].value

        exp = get_exp()

        outcomes = exp.outcomes

        widgets['outcome'].options = [('_'.join(outcome), outcome) for outcome in outcomes]
        if len(outcomes) > 0:
            if previous_value in outcomes:
                widgets['outcome'].value = previous_value
                populate_read_ids(None)
            else:
                widgets['outcome'].value = widgets['outcome'].options[0][1]
        else:
            widgets['outcome'].value = None

    @output.capture()
    def populate_read_ids(change):
        exp = get_exp()

        df = exp.filtered_cell_outcomes

        if exp is None:
            return

        if by_outcome:
            outcome = widgets['outcome'].value
            if outcome is None:
                qnames = []
            else:
                category, subcategory = outcome
                query = 'category == @category and subcategory == @subcategory'
                if 'guide_mismatch' in df.columns:
                    query += ' and guide_mismatch == -1'
                right_outcome = df.query(query)

                if 'original_name' in df.columns:
                    qname_key = 'original_name'
                else:
                    qname_key = 'query_name'

                qnames = right_outcome[qname_key].values[:200]

        else:
            qnames = df['original_name'].values[:200]

        widgets['read_id'].options = qnames

        if len(qnames) > 0:
            widgets['read_id'].value = qnames[0]
            widgets['read_id'].index = 0
        else:
            widgets['read_id'].value = None
            
    if by_outcome:
        populate_outcomes({'name': 'initial'})

    populate_read_ids({'name': 'initial'})

    if by_outcome:
        widgets['outcome'].observe(populate_read_ids, names='value')
        widgets['guide'].observe(populate_outcomes, names='value')
    else:
        widgets['guide'].observe(populate_read_ids, names='value')

    @output.capture(clear_output=clear_output)
    def plot(guide, read_id, **plot_kwargs):
        exp = get_exp()

        if exp is None:
            return

        if by_outcome:
            als = exp.get_read_alignments(read_id, fn_key='filtered_cell_bam_by_name', outcome=plot_kwargs['outcome'])
        else:
            als = exp.get_read_alignments(read_id)

        if als is None:
            return None

        l = pool.categorizer(als, exp.target_info)
        category, subcategory, details, outcome = l.categorize()
        if widgets['relevant'].value:
            als = l.relevant_alignments

        diagram = visualize.ReadDiagram(als, exp.target_info,
                                        max_qual=exp.max_qual,
                                        **plot_kwargs,
                                       )
        fig = diagram.fig

        title = f'{l.query_name} {category} {subcategory} {details}'
        fig.axes[0].set_title(title)

        if widgets['draw_sequence'].value:
            print(exp.group, exp.sample_name)
            print(als[0].query_name)
            print(als[0].get_forward_sequence())

        return fig

    all_kwargs = {**{k: ipywidgets.fixed(v) for k, v in kwargs.items()}, **widgets}

    interactive = ipywidgets.interactive(plot, **all_kwargs)
    interactive.update()

    def make_row(keys):
        return ipywidgets.HBox([widgets[k] if k in widgets else non_widgets[k] for k in keys])

    if by_outcome:
        top_row_keys = ['guide', 'outcome', 'read_id']
    else:
        top_row_keys = ['guide', 'read_id']

    @output.capture(clear_output=False)
    def save(change):
        fig = interactive.result
        fn = non_widgets['file_name'].value
        fig.savefig(fn, bbox_inches='tight')

    non_widgets['save'].on_click(save)

    layout = ipywidgets.VBox(
        [make_row(top_row_keys),
         make_row([k for k, d in toggles]),
         make_row(['save', 'file_name']),
         interactive.children[-1],
         output,
        ],
    )

    return layout

class MergedPools(PooledScreen):
    def __init__(self, base_dir, name, groups, progress=None):
        super().__init__(base_dir, name, progress=progress)

        self.groups = groups
        self.pools = {group: PooledScreen(base_dir, group, progress) for group in groups}

    @memoized_property
    def R2_read_length(self):
        pool = next(iter(self.pools.values()))
        return pool.R2_read_length

    def merge_outcome_counts(self):
        all_counts = {group: pool.outcome_counts_df(True) for group, pool in self.pools.items()}
        all_counts = pd.concat(all_counts, axis=1).fillna(0).astype(int)
        merged_counts = all_counts.sum(axis='columns', level=[1, 2])

        sparse_counts = scipy.sparse.coo_matrix(merged_counts.values)
        scipy.sparse.save_npz(self.fns['collapsed_outcome_counts'], sparse_counts)

        merged_counts.sum(axis=1).to_csv(self.fns['collapsed_total_outcome_counts'], header=False)
        
    def merge_templated_insertion_details(self):
        all_counts = defaultdict(Counter)
        
        def add_to_merged(key, group):
            fields = key.split('/')

            if len(fields) != 4:
                return
            
            all_counts[key].update(dict(zip(group['values'], group['counts'])))
        
        for pool_name, pool in self.pools.items():
            with h5py.File(pool.fns['filtered_templated_insertion_details']) as pool_f:
                pool_f.visititems(add_to_merged)
        
        with h5py.File(self.fns['filtered_templated_insertion_details'], 'w') as merged_f:
            for key, counts in self.progress(all_counts.items(), total=len(all_counts)):
                if len(counts) == 0:
                    values = np.array([], dtype=int)
                    counts = np.array([], dtype=int)
                else:
                    values = np.array(sorted(counts))
                    counts = np.array([counts[v] for v in values])

                merged_f.create_dataset(f'{key}/values', data=values)
                merged_f.create_dataset(f'{key}/counts', data=counts)

def get_pool(base_dir, group, progress=None):
    group_dir = Path(base_dir) / 'results' / group
    sample_sheet_fn = group_dir / 'sample_sheet.yaml'
    if sample_sheet_fn.exists():
        sample_sheet = yaml.safe_load(sample_sheet_fn.read_text())
        pooled = sample_sheet.get('pooled', False)
        if pooled:
            if sample_sheet.get('gDNA', False):
                pool = PooledScreenNoUMI(base_dir, group, progress=progress)
            elif 'pools_to_merge' in sample_sheet:
                pool = MergedPools(base_dir, group, groups=sample_sheet['pools_to_merge'], progress=progress)
            else:
                pool = PooledScreen(base_dir, group, progress=progress)
        else:
            pool = None
    else:
        pool = None

    return pool

def get_all_pools(base_dir=Path.home() / 'projects' / 'ddr', progress=None):
    group_dirs = [p for p in (Path(base_dir) / 'results').iterdir() if p.is_dir()]

    pools = {}

    for group_dir in group_dirs:
        name = group_dir.name
        pool = get_pool(base_dir, name, progress=progress)
        if pool is not None:
            pools[name] = pool

    return pools