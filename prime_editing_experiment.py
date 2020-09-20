import shutil
import itertools
import gzip
from collections import defaultdict, Counter

import pysam
import pandas as pd
import h5py
import numpy as np

import hits.visualize
from hits import fastq, utilities, adapters
from knock_knock import experiment, visualize, svg
from ddr import prime_editing_layout, pooled_layout

from hits.utilities import memoized_property

class PrimeEditingExperiment(experiment.Experiment):
    def __init__(self, base_dir, group, name, **kwargs):
        super().__init__(base_dir, group, name, **kwargs)

        self.read_types = [
            'trimmed',
        ]

        self.default_read_type = 'trimmed'

        self.layout_mode = 'amplicon'
        self.max_qual = 41

        self.fns['fastq'] = self.data_dir / self.description['fastq_fn']
        self.fns['templated_insertion_details'] = self.results_dir / 'templated_insertion_details.hdf5'

        self.outcome_fn_keys = ['outcome_list']

        self.length_to_store_unknown = None
        self.length_plot_smooth_window = 0
        self.x_tick_multiple = 50

        label_offsets = {}
        for (seq_name, feature_name), feature in self.target_info.features.items():
            if feature.feature.startswith('donor_insertion'):
                label_offsets[feature_name] = 2
                self.target_info.features_to_show.add((seq_name, feature_name))
            elif feature.feature.startswith('donor_deletion'):
                label_offsets[feature_name] = 3
                self.target_info.features_to_show.add((seq_name, feature_name))
            elif feature_name.startswith('HA_RT'):
                label_offsets[feature_name] = 1

        label_offsets[f'{self.target_info.primary_sgRNA}_PAM'] = 2

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

    @memoized_property
    def categorizer(self):
        return prime_editing_layout.Layout

    @memoized_property
    def max_relevant_length(self):
        outcomes = self.outcome_iter()
        return max(outcome.length for outcome in outcomes)
               
    def trim_reads(self):
        # Trim a random length barcode from the beginning by searching for the expected starting sequence.
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
            reads = fastq.reads(self.fns['fastq'])
            #reads = itertools.islice(reads, 1000)
            for read in self.progress(reads, desc='Trimming reads'):
                try:
                    start = read.seq.index(prefix, 0, 20)
                except ValueError:
                    start = 0

                end = adapters.trim_by_local_alignment(adapters.truseq_R2_rc, read.seq)
                trimmed_fh.write(str(read[start:end]))

    def process(self, stage):
        try:
            if stage == 'preprocess':
                self.trim_reads()
            elif stage == 'align':
                self.generate_alignments(read_type='trimmed')
                self.generate_supplemental_alignments_with_STAR(read_type='trimmed', min_length=20)
                self.combine_alignments(read_type='trimmed')
            elif stage == 'categorize':
                self.categorize_outcomes(read_type='trimmed')
                self.count_read_lengths()
                self.extract_templated_insertion_info()
            elif stage == 'visualize':
                lengths_fig = self.length_distribution_figure()
                lengths_fig.savefig(self.fns['lengths_figure'], bbox_inches='tight')
                self.generate_all_outcome_length_range_figures()
                svg.decorate_outcome_browser(self)
                self.generate_all_outcome_example_figures(num_examples=5)
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
                    layout = self.categorizer(als, self.target_info, mode=self.layout_mode)
                    category, subcategory, details, outcome = layout.categorize()
                    if outcome is not None:
                        details = str(outcome.perform_anchor_shift(self.target_info.anchor))
                except:
                    print(self.name, name)
                    raise
                
                outcomes[category, subcategory].append(name)

                if layout.seq is None:
                    length = 0
                else:
                    length = layout.inferred_amplicon_length

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

    @memoized_property
    def qname_to_inferred_length(self):
        qname_to_inferred_length = {}
        for outcome in self.outcome_iter():
            qname_to_inferred_length[outcome.query_name] = outcome.length

        return qname_to_inferred_length

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