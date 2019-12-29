import shutil
import gzip
from collections import defaultdict

import pysam
import pandas as pd

from hits import fastq, utilities
from knock_knock import experiment, read_outcome, visualize, svg
from ddr import prime_editing_layout

class AmpliconExperiment(experiment.Experiment):
    def __init__(self, base_dir, group, name, **kwargs):
        super().__init__(base_dir, group, name, **kwargs)

        self.read_types = [
            'trimmed',
        ]

        self.layout_mode = 'amplicon'
        self.layout_module = prime_editing_layout
        self.max_qual = 41

        self.fns['fastq'] = self.data_dir / self.description['fastq_fn']

        self.outcome_fn_keys = ['outcome_list']

        self.max_relevant_length = 325
        self.length_to_store_unknown = None
        self.length_plot_smooth_window = 0
        self.x_tick_multiple = 50
               
    def preprocess(self):
        trimmed_fn = self.fns_by_read_type['fastq']['trimmed']
        with gzip.open(trimmed_fn, 'wt', compresslevel=1) as trimmed_fh:
            reads = fastq.reads(self.fns['fastq'])
            for read in self.progress(reads, desc='Trimming reads'):
                try:
                    start = read.seq.index('GCCTTT', 0, 20)
                    trimmed_fh.write(str(read[start:]))
                except ValueError:
                    pass

    def process(self, stage):
        if stage == 'align':
            self.preprocess()
            self.generate_alignments(read_type='trimmed')
            self.generate_supplemental_alignments(read_type='trimmed', min_length=20)
            self.combine_alignments(read_type='trimmed')
        elif stage == 'categorize':
            self.categorize_outcomes(read_type='trimmed')
            self.count_read_lengths()
        elif stage == 'visualize':
            lengths_fig = self.length_distribution_figure()
            lengths_fig.savefig(self.fns['lengths_figure'], bbox_inches='tight')
            self.generate_all_outcome_length_range_figures()
            svg.decorate_outcome_browser(self)
            self.generate_all_outcome_example_figures(num_examples=5, split_at_indels=True, flip_donor=True, highlight_SNPs=True)

    def categorize_outcomes(self, fn_key='bam_by_name', read_type=None):
        if self.fns['outcomes_dir'].is_dir():
            shutil.rmtree(str(self.fns['outcomes_dir']))
            
        self.fns['outcomes_dir'].mkdir()

        outcomes = defaultdict(list)

        with self.fns['outcome_list'].open('w') as fh:
            alignment_groups = self.alignment_groups(fn_key, read_type=read_type)

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

                if layout.seq is None:
                    length = 0
                else:
                    length = len(layout.seq)

                outcome = read_outcome.Outcome(name, length, category, subcategory, details)
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

    def alignment_groups(self, fn_key='bam_by_name', outcome=None, read_type='trimmed'):
        groups = super().alignment_groups(fn_key=fn_key, outcome=outcome, read_type=read_type)
        return groups

    generate_supplemental_alignments = experiment.IlluminaExperiment.generate_supplemental_alignments

    def generate_length_range_figures(self, outcome=None, num_examples=1):
        by_length = defaultdict(lambda: utilities.ReservoirSampler(num_examples))

        al_groups = self.alignment_groups(outcome=outcome, read_type='trimmed')
        for name, als in al_groups:
            length = als[0].query_length
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
            diagrams = self.alignment_groups_to_diagrams(sampler.sample, num_examples=num_examples, split_at_indels=True, flip_donor=True, highlight_SNPs=True)
            im = visualize.make_stacked_Image(diagrams, titles='')
            fn = fns['length_range_figure'](length, length)
            im.save(fn)
