import shutil
import gzip
from collections import defaultdict, Counter

import h5py
import numpy as np

import hits.visualize
from hits import fastq, utilities, adapters
from knock_knock import experiment, svg
from repair_seq import prime_editing_layout, pooled_layout

from hits.utilities import memoized_property

class PrimeEditingExperiment(experiment.Experiment):
    def __init__(self, base_dir, group, sample_name, **kwargs):
        super().__init__(base_dir, group, sample_name, **kwargs)

        self.read_types = [
            'trimmed',
        ]

        self.layout_mode = 'amplicon'
        self.max_qual = 41

        self.fns['templated_insertion_details'] = self.results_dir / 'templated_insertion_details.hdf5'

        self.outcome_fn_keys = ['outcome_list']

        self.length_plot_smooth_window = 0
        self.x_tick_multiple = 50

        label_offsets = {}
        for (seq_name, feature_name), feature in self.target_info.features.items():
            if feature.feature.startswith('donor_insertion'):
                label_offsets[feature_name] = 2
                #self.target_info.features_to_show.add((seq_name, feature_name))
            elif feature.feature.startswith('donor_deletion'):
                label_offsets[feature_name] = 3
                #self.target_info.features_to_show.add((seq_name, feature_name))
            elif feature_name.startswith('HA_RT'):
                label_offsets[feature_name] = 1

        label_offsets[f'{self.target_info.primary_sgRNA}_PAM'] = 2
        other_sgRNAs = [sgRNA for sgRNA in self.target_info.sgRNAs if sgRNA != self.target_info.primary_sgRNA]
        for sgRNA in other_sgRNAs:
            label_offsets[f'{sgRNA}_PAM'] = 1

        self.target_info.features_to_show.update(set(self.target_info.PAM_features))
        
        self.diagram_kwargs.update(draw_sequence=True,
                                   flip_target=self.target_info.sequencing_direction == '-',
                                   flip_donor=self.target_info.sgRNA_feature.strand == self.target_info.sequencing_direction,
                                   highlight_SNPs=True,
                                   center_on_primers=True,
                                   split_at_indels=True,
                                   force_left_aligned=False,
                                   label_offsets=label_offsets,
                                  )

        self.read_types = [
            'trimmed',
            'trimmed_by_name',
            'nonredundant',
        ]

    def __repr__(self):
        return f'PrimeEditingExperiment: sample_name={self.sample_name}, group={self.group}, base_dir={self.base_dir}'

    @property
    def default_read_type(self):
        return 'trimmed_by_name'

    @property
    def preprocessed_read_type(self):
        return 'trimmed_by_name'

    @property
    def read_types_to_align(self):
        return ['nonredundant']

    @memoized_property
    def categorizer(self):
        return prime_editing_layout.Layout

    @memoized_property
    def max_relevant_length(self):
        outcomes = self.outcome_iter()
        return max(outcome.inferred_amplicon_length for outcome in outcomes)
    
    def make_nonredundant_sequence_fastq(self):
        pass
               
    def trim_reads(self):
        ''' Trim a random length barcode from the beginning by searching for the expected starting sequence.
        '''
        fastq_fn = self.data_dir / self.description['fastq_fn']

        # Standardizing names is important for sorting.
        reads = fastq.reads(fastq_fn, standardize_names=True)

        ti = self.target_info

        if ti.sequencing_direction == '+':
            start = ti.sequencing_start.start
            prefix = ti.target_sequence[start:start + 6]
        else:
            end = ti.sequencing_start.end
            prefix = utilities.reverse_complement(ti.target_sequence[end - 5:end + 1])

        prefix = prefix.upper()

        trimmed_fn = self.fns_by_read_type['fastq']['trimmed']
        with gzip.open(trimmed_fn, 'wt', compresslevel=1) as trimmed_fh:
            for read in self.progress(reads, desc='Trimming reads'):
                try:
                    start = read.seq.index(prefix, 0, 20)
                except ValueError:
                    start = 0

                end = adapters.trim_by_local_alignment(adapters.truseq_R2_rc, read.seq)
                trimmed_fh.write(str(read[start:end]))

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

                self.generate_outcome_counts()
                self.generate_read_lengths()

                self.extract_templated_insertion_info()

                self.record_sanitized_category_names()

            elif stage == 'visualize':
                lengths_fig = self.length_distribution_figure()
                lengths_fig.savefig(self.fns['lengths_figure'], bbox_inches='tight')
                self.generate_all_outcome_length_range_figures()
                svg.decorate_outcome_browser(self)
                self.generate_all_outcome_example_figures(num_examples=5)
        except:
            print(self.group, self.sample_name)
            raise

    @memoized_property
    def qname_to_inferred_length(self):
        qname_to_inferred_length = {}
        for outcome in self.outcome_iter():
            qname_to_inferred_length[outcome.query_name] = outcome.inferred_amplicon_length

        return qname_to_inferred_length

    def generate_length_range_figures(self, outcome=None, num_examples=1):
        by_length = defaultdict(lambda: utilities.ReservoirSampler(num_examples))

        #al_groups = self.alignment_groups(outcome=outcome, read_type='trimmed')
        al_groups = self.alignment_groups(outcome=outcome)
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
            im = hits.visualize.make_stacked_Image([d.fig for d in diagrams])
            fn = fns['length_range_figure'](length, length)
            im.save(fn)

    def extract_templated_insertion_info(self):
        fields = prime_editing_layout.LongTemplatedInsertionOutcome.int_fields
        
        lists = defaultdict(list)

        with open(self.fns['outcome_list']) as outcomes_fh:
            for line in outcomes_fh:
                outcome = self.final_Outcome.from_line(line)
            
                if outcome.category == 'unintended donor integration':
                    insertion_outcome = prime_editing_layout.LongTemplatedInsertionOutcome.from_string(outcome.details)
                    
                    for field in fields: 
                        value = getattr(insertion_outcome, field)
                        key = f'{outcome.category}/{outcome.subcategory}/{field}'
                        lists[key].append(value)
                            
        with h5py.File(self.fns['templated_insertion_details'], 'w') as hdf5_file:
            cat_and_subcats = {key.rsplit('/', 1)[0] for key in lists}
            read_length = 258
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

        if pooled_layout.NAN_INT in counts:
            counts.pop(pooled_layout.NAN_INT)

        if len(counts) == 0:
            xs = np.array([])
        else:
            xs = np.arange(min(counts), max(counts) + 1)

        ys = np.array([counts[x] for x in xs])

        return xs, ys