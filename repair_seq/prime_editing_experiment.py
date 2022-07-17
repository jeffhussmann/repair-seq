import gzip
from collections import defaultdict, Counter

import h5py
import numpy as np

from hits import utilities
from knock_knock import illumina_experiment, svg
from repair_seq import prime_editing_layout, twin_prime_layout, pooled_layout

from hits.utilities import memoized_property

class PrimeEditingExperiment(illumina_experiment.IlluminaExperiment):
    def __init__(self, base_dir, group, sample_name, **kwargs):
        super().__init__(base_dir, group, sample_name, **kwargs)

    @property
    def read_types_to_align(self):
        return ['nonredundant']
        #return ['trimmed_by_name']

    @memoized_property
    def categorizer(self):
        return prime_editing_layout.Layout

    @memoized_property
    def max_relevant_length(self):
        outcomes = self.outcome_iter()
        longest_seen = max(outcome.inferred_amplicon_length for outcome in outcomes)
        return min(longest_seen, 600)
    
    def make_nonredundant_sequence_fastq(self):
        # This is overloaded by ArrayedExperiment.
        fn = self.fns_by_read_type['fastq']['nonredundant']
        with gzip.open(fn, 'wt', compresslevel=1) as fh:
            for read in self.reads_by_type(self.preprocessed_read_type):
                fh.write(str(read))

    def visualize(self):
        lengths_fig = self.length_distribution_figure()
        lengths_fig.savefig(self.fns['lengths_figure'], bbox_inches='tight')
        self.generate_all_outcome_length_range_figures()
        svg.decorate_outcome_browser(self)
        self.generate_all_outcome_example_figures(num_examples=5)

    def process_old(self, stage):
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
                self.visualize()
        except:
            print(self.group, self.sample_name)
            raise

    def alignment_groups_to_diagrams(self, alignment_groups, num_examples, **diagram_kwargs):
        subsample = utilities.reservoir_sample(alignment_groups, num_examples)

        for qname, als in subsample:
            layout = self.categorizer(als, self.target_info, mode=self.layout_mode)

            layout.categorize()
            
            try:
                diagram = layout.plot(title='', **diagram_kwargs)
            except:
                print(self.sample_name, qname)
                raise
                
            yield diagram

    def extract_templated_insertion_info(self):
        fields = prime_editing_layout.LongTemplatedInsertionOutcome.int_fields
        
        lists = defaultdict(list)

        for outcome in self.outcome_iter():
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

class TwinPrimeExperiment(PrimeEditingExperiment):
    def __init__(self, base_dir, group, sample_name, **kwargs):
        super().__init__(base_dir, group, sample_name, **kwargs)

    @memoized_property
    def categorizer(self):
        return twin_prime_layout.Layout