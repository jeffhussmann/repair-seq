import shutil
import gzip
from collections import defaultdict, Counter

import knock_knock.experiment
import ddr.pooled_screen
import ddr.pooled_layout

from hits import utilities, adapters, fastq
from knock_knock import svg, visualize

import pandas as pd
import pysam
import h5py
import numpy as np

memoized_property = utilities.memoized_property

class Experiment(knock_knock.experiment.Experiment):
    def __init__(self, base_dir, group, name, **kwargs):
        super().__init__(base_dir, group, name, **kwargs)

        self.read_types = [
            'trimmed',
        ]

        self.layout_mode = 'amplicon'
        self.max_qual = 41

        self.fns['fastq'] = self.data_dir / self.description['fastq_fn']
        self.fns['templated_insertion_details'] = self.fns['dir'] / 'templated_insertion_details.hdf5'

        self.outcome_fn_keys = ['outcome_list']

        self.length_to_store_unknown = None
        self.length_plot_smooth_window = 0
        self.x_tick_multiple = 50

        self.target_info.features_to_show.update(set(self.target_info.PAM_features))

        self.diagram_kwargs = dict(draw_sequence=True,
                                   flip_target=self.target_info.sequencing_direction == '-',
                                   flip_donor=True,
                                   highlight_SNPs=True,
                                   center_on_primers=True,
                                   split_at_indels=True,
                                   ref_centric=True,
                                   force_left_aligned=False,
                                   features_to_show=self.target_info.features_to_show,
                                  )

        self.layout_mode = 'no_UMI'

    @memoized_property
    def layout_module(self):
        return ddr.pooled_layout

    @memoized_property
    def max_relevant_length(self):
        outcomes = self.outcome_iter()
        return max(outcome.length for outcome in outcomes)

    def preprocess(self):
        trimmed_fn = self.fns_by_read_type['fastq']['trimmed']
        with gzip.open(trimmed_fn, 'wt', compresslevel=1) as trimmed_fh:
            reads = fastq.reads(self.fns['fastq'])
            for read in self.progress(reads, desc='Trimming reads'):
                end = adapters.trim_by_local_alignment(adapters.truseq_R2_rc, read.seq)
                trimmed_read = read[:end]
                if len(trimmed_read) > 0:
                    trimmed_fh.write(str(trimmed_read))

    def process(self, stage):
        try:
            if stage == 'align':
                self.preprocess()
                self.generate_alignments(read_type='trimmed')
                self.generate_supplemental_alignments(read_type='trimmed', min_length=20)
                self.combine_alignments(read_type='trimmed')
            elif stage == 'categorize':
                self.categorize_outcomes(read_type='trimmed')
                self.count_read_lengths()
                self.make_outcome_counts()
            elif stage == 'visualize':
                lengths_fig = self.length_distribution_figure()
                lengths_fig.savefig(self.fns['lengths_figure'], bbox_inches='tight')
                self.generate_all_outcome_length_range_figures()
                svg.decorate_outcome_browser(self)
                self.generate_all_outcome_example_figures(num_examples=5,
                                                          split_at_indels=True,
                                                          flip_donor=True,
                                                          flip_target=self.target_info.sequencing_start.strand == '-',
                                                          highlight_SNPs=True,
                                                         )
        except:
            print(self.group, self.name)
            raise

    def categorize_outcomes(self, fn_key='bam_by_name', read_type=None):
        if self.fns['outcomes_dir'].is_dir():
            shutil.rmtree(str(self.fns['outcomes_dir']))
            
        self.fns['outcomes_dir'].mkdir()

        outcomes = defaultdict(list)

        with self.fns['outcome_list'].open('w') as fh:
            alignment_groups = self.alignment_groups(fn_key, read_type=read_type)
            #alignment_groups = itertools.islice(alignment_groups, 1000)

            if read_type is None:
                description = 'Categorizing reads'
            else:
                description = f'Categorizing {read_type} reads'

            for name, als in self.progress(alignment_groups, desc=description):
                try:
                    layout = self.layout_module.Layout(als, self.target_info, mode=self.layout_mode)
                    category, subcategory, details, outcome = layout.categorize()
                    if outcome is not None:
                        details = str(outcome.perform_anchor_shift(self.target_info.anchor))
                except:
                    print(self.name, name)
                    raise
                
                outcomes[category, subcategory].append(name)

                length = len(layout.seq)

                outcome = self.final_Outcome(name, length, category, subcategory, details)
                fh.write(f'{outcome}\n')

        counts = {description: len(names) for description, names in outcomes.items()}
        pd.Series(counts).to_csv(self.fns['outcome_counts'], sep='\t', header=False)

        # To make plotting easier, for each outcome, make a file listing all of
        # qnames for the outcome and a bam file (sorted by name) with all of the
        # alignments for these qnames.

        qname_to_outcome = {}
        bam_fhs = {}

        full_bam_fn = self.fns_by_read_type[fn_key][read_type]

        with pysam.AlignmentFile(full_bam_fn) as full_bam_fh:
        
            for outcome, qnames in outcomes.items():
                outcome_fns = self.outcome_fns(outcome)

                # This shouldn't be necessary due to rmtree of parent directory above
                # but empirically sometimes is.
                if outcome_fns['dir'].is_dir():
                    shutil.rmtree(str(outcome_fns['dir']))

                outcome_fns['dir'].mkdir()

                bam_fn = outcome_fns['bam_by_name'][read_type]
                bam_fhs[outcome] = pysam.AlignmentFile(bam_fn, 'wb', template=full_bam_fh)
                
                with outcome_fns['query_names'].open('w') as fh:
                    for qname in qnames:
                        qname_to_outcome[qname] = outcome
                        fh.write(qname + '\n')
            
            for al in full_bam_fh:
                if al.query_name in qname_to_outcome:
                    outcome = qname_to_outcome[al.query_name]
                    bam_fhs[outcome].write(al)

        for outcome, fh in bam_fhs.items():
            fh.close()

    def get_read_alignments(self, read_id, fn_key=None, outcome=None, read_type='trimmed'):
        return super().get_read_alignments(read_id, fn_key='bam_by_name', outcome=outcome, read_type=read_type)

    def get_read_layout(self, read_id, fn_key=None, outcome=None, read_type='trimmed'):
        return super().get_read_layout(read_id, fn_key='bam_by_name', outcome=outcome, read_type=read_type)

    def get_read_diagram(self, read_id, **kwargs):
        return super().get_read_diagram(read_id, read_type='trimmed', **kwargs)

    def alignment_groups(self, fn_key='bam_by_name', outcome=None, read_type='trimmed'):
        groups = super().alignment_groups(fn_key=fn_key, outcome=outcome, read_type=read_type)
        return groups

    @memoized_property
    def qname_to_inferred_length(self):
        qname_to_inferred_length = {}
        for outcome in self.outcome_iter():
            qname_to_inferred_length[outcome.query_name] = outcome.length

        return qname_to_inferred_length

    generate_supplemental_alignments = knock_knock.experiment.IlluminaExperiment.generate_supplemental_alignments

    def generate_length_range_figures(self, outcome=None, num_examples=1):
        by_length = defaultdict(lambda: utilities.ReservoirSampler(num_examples))

        al_groups = self.alignment_groups(outcome=outcome, read_type='trimmed')
        for name, als in al_groups:
            length = self.qname_to_inferred_length[name]
            by_length[length].add((name, als))
        
        if outcome is None:
            fns = self.fns
        else:
            fns = self.outcome_fns(outcome)

        fig_dir = fns['length_ranges_dir']
            
        if fig_dir.is_dir():
            shutil.rmtree(str(fig_dir))
        fig_dir.mkdir()

        if outcome is not None:
            description = ': '.join(outcome)
        else:
            description = 'Generating length-specific diagrams'

        items = self.progress(by_length.items(), desc=description, total=len(by_length))

        for length, sampler in items:
            diagrams = self.alignment_groups_to_diagrams(sampler.sample,
                                                         num_examples=num_examples,
                                                         **self.diagram_kwargs,
                                                        )
            im = visualize.make_stacked_Image(diagrams, titles='')
            fn = fns['length_range_figure'](length, length)
            im.save(fn)

    def make_outcome_counts(self):
        counts = Counter()

        for outcome in self.outcome_iter():
            counts[True, outcome.category, outcome.subcategory, outcome.details] += 1

        counts = pd.Series(counts).sort_values(ascending=False)
        counts.to_csv(self.fns['outcome_counts'], sep='\t', header=False)

    @memoized_property
    def outcome_counts(self):
        counts = pd.read_table(self.fns['outcome_counts'],
                               header=None,
                               index_col=[0, 1, 2, 3],
                               squeeze=True,
                               na_filter=False,
                              )
        counts.index.names = ['perfect_guide', 'category', 'subcategory', 'details']
        return counts

    @memoized_property
    def outcomes(self):
        ''' (category, subcategory) tuples in descending order by total count '''
        totals = self.outcome_counts[True].groupby(by=['category', 'subcategory']).sum().sort_values(ascending=False)
        return totals.index.values

    @memoized_property
    def outcome_fractions(self):
        outcome_counts = self.outcome_counts[True].copy()

        if 'wild type' in outcome_counts:
            total_wild_type = outcome_counts['wild type'].sum()
            outcome_counts = outcome_counts.drop('wild type', level=0)
        else:
            total_wild_type = 0
        outcome_counts.loc['wild type', 'collapsed', 'n/a'] = total_wild_type

        if ('genomic insertion', 'hg19') in outcome_counts:
            total_genomic = outcome_counts['genomic insertion', 'hg19'].sum()
            outcome_counts = outcome_counts.drop(('genomic insertion', 'hg19'))
        else:
            total_genomic = 0
        outcome_counts.loc['genomic insertion', 'hg19', 'collapsed'] = total_genomic

        outcome_counts = outcome_counts.drop('nonspecific amplification', errors='ignore')

        outcome_counts = outcome_counts.sort_values(ascending=False)

        outcome_fractions = outcome_counts / outcome_counts.sum()

        return outcome_fractions

    def extract_templated_insertion_info(self):
        fields = self.layout_module.LongTemplatedInsertionOutcome.int_fields
        
        lists = defaultdict(list)

        for outcome in self.outcome_iter():
            if outcome.category in ['donor insertion', 'genomic insertion']:
                insertion_outcome = self.layout_module.LongTemplatedInsertionOutcome.from_string(outcome.details)
                
                for field in fields: 
                    value = getattr(insertion_outcome, field)
                    key = f'{outcome.category}/{outcome.subcategory}/{field}'
                    lists[key].append(value)
                            
        with h5py.File(self.fns['templated_insertion_details'], 'w') as hdf5_file:
            cat_and_subcats = {key.rsplit('/', 1)[0] for key in lists}
            read_length = 250
            for cat_and_subcat in cat_and_subcats:
                left_key = f'{cat_and_subcat}/left_insertion_query_bound'
                right_key = f'{cat_and_subcat}/right_insertion_query_bound'

                lengths = []

                for left, right in zip(lists[left_key], lists[right_key]):
                    if right == read_length - 1:
                        length = read_length
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

    def templated_insertion_details(self, category, subcategories, field):
        counts = Counter()

        if isinstance(subcategories, str):
            subcategories = [subcategories]

        with h5py.File(self.fns[f'templated_insertion_details']) as f:
            for subcategory in subcategories:
                group = f'{category}/{subcategory}/{field}'
                if group in f:
                    counts.update(dict(zip(f[group]['values'], f[group]['counts'])))

        if ddr.pooled_layout.NAN_INT in counts:
            counts.pop(ddr.pooled_layout.NAN_INT)

        if len(counts) == 0:
            xs = np.array([])
        else:
            xs = np.arange(min(counts), max(counts) + 1)

        ys = np.array([counts[x] for x in xs])

        return xs, ys
