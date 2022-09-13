import gzip
from collections import defaultdict, Counter

import h5py
import matplotlib.pyplot as plt
import numpy as np

import hits.visualize
from hits import utilities
from knock_knock import illumina_experiment, pegRNAs
from repair_seq import prime_editing_layout, twin_prime_layout, Bxb1_layout, pooled_layout

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

    def generate_figures(self):
        lengths_fig = self.length_distribution_figure()
        lengths_fig.savefig(self.fns['lengths_figure'], bbox_inches='tight')
        self.generate_all_outcome_length_range_figures()
        self.generate_outcome_browser()
        self.generate_all_outcome_example_figures(num_examples=5)

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
    @memoized_property
    def categorizer(self):
        return twin_prime_layout.Layout

    def plot_extension_chain_edges(self):
        ti = self.target_info
        features = ti.annotated_and_inferred_features

        edge_distributions = defaultdict(Counter)
        joint_distribution = Counter()

        for outcome in self.outcome_iter():
            if outcome.category.startswith('unintended rejoining'):
                ur_outcome = twin_prime_layout.UnintendedRejoiningOutcome.from_string(outcome.details)

                for side_description in outcome.subcategory.split(', '):
                    side, description = side_description.split(' ', 1)
                    edge_distributions[side, description][ur_outcome.edges[side]] += 1

                joint_key = (outcome.subcategory, outcome.details)
                joint_distribution[joint_key] += 1

        # Common parameters.
        ref_bar_height = 0.02
        feature_height = 0.03
        gap_between_refs = 0.01

        figsize = (16, 4)
        x_lims = (-200, 250)

        # not RT'ed
        subcategory_key = "not RT'ed"

        fig, axs = plt.subplots(1, 2, figsize=figsize)

        for ax, side in zip(axs, ['left', 'right']):
            counts = edge_distributions[side, subcategory_key]

            if len(counts) == 0:
                continue

            xs = np.arange(min(counts), max(counts) + 1)
            ys = [counts[x] for x in xs]

            ax.plot(xs, ys, '.-', color='black', alpha=0.5)

            pegRNA_name = ti.pegRNA_names_by_side_of_read[side]
            protospacer_name = pegRNAs.protospacer_name(pegRNA_name)

            PBS_name = ti.PBS_names_by_side_of_read[side]
            PBS = features[ti.target, PBS_name]

            other_pegRNA_name = ti.pegRNA_names_by_side_of_read[twin_prime_layout.other_side[side]]
            other_PBS_name = ti.PBS_names_by_side_of_read[twin_prime_layout.other_side[side]]
            other_protospacer_name = pegRNAs.protospacer_name(other_pegRNA_name)
            PAM_name = f'{protospacer_name}_PAM'
            other_PAM_name = f'{other_protospacer_name}_PAM'

            colors = {
                protospacer_name: hits.visualize.apply_alpha(ti.pegRNA_name_to_color[pegRNA_name], 0.5),
                other_protospacer_name: hits.visualize.apply_alpha(ti.pegRNA_name_to_color[other_pegRNA_name], 0.5),
                PAM_name: ti.pegRNA_name_to_color[pegRNA_name],
                other_PAM_name: ti.pegRNA_name_to_color[other_pegRNA_name],
            }

            for primer_name in ti.primer_names:
                colors[primer_name] = 'lightgrey'

            # By definition, the edge of the PBS adjacent to the nick in the target
            # for this side's pegRNA is zero in the coordinate system.

            y_start = -0.1

            feature_names = [
                protospacer_name,
                other_protospacer_name,
                PBS_name, other_PBS_name,
                PAM_name,
                other_PAM_name
            ] + ti.primer_names

            for feature_name in feature_names:
                feature = features[ti.target, feature_name]
                
                # Moving towards the other nicks is moving
                # forward in the coordinate system.
                if PBS.strand == '+':
                    start, end = feature.start - PBS.end - 0.5, feature.end - PBS.end + 0.5
                else:
                    start, end = PBS.start - feature.end - 0.5, PBS.start - feature.start + 0.5

                if 'PBS' in feature_name:
                    height = 0.015
                else:
                    height = 0.03
                
                ax.axvspan(start, end,
                           y_start, y_start - height,
                           facecolor=colors.get(feature_name, feature.attribute['color']),
                           clip_on=False,
                          )

            ax.axvspan(x_lims[0], x_lims[1], y_start, y_start + ref_bar_height, facecolor='C0', clip_on=False)

            for cut_after_name, cut_after in ti.cut_afters.items():
                if PBS.strand == '+':
                    x = cut_after - PBS.end
                else:
                    x = PBS.start - cut_after

                name, strand = cut_after_name.rsplit('_', 1)

                ref_y = y_start + 0.5 * ref_bar_height
                cut_y_bottom = ref_y - feature_height
                cut_y_middle = ref_y
                cut_y_top = ref_y + feature_height

                if (strand == '+' and ti.sequencing_direction == '+') or (strand == '-' and ti.sequencing_direction == '-'):
                    ys = [cut_y_middle, cut_y_top]
                elif (strand == '-' and ti.sequencing_direction == '+') or (strand == '+' and ti.sequencing_direction == '-'):
                    ys = [cut_y_bottom, cut_y_middle]
                else:
                    raise ValueError

                ax.plot([x, x],
                         ys,
                         '-',
                         linewidth=1,
                         color='black',
                         solid_capstyle='butt',
                         zorder=10,
                         transform=ax.get_xaxis_transform(),
                         clip_on=False,
                        )

            ax.set_xlim(*x_lims)
            ax.set_ylim(0)

            ax.set_title(f'{side} {subcategory_key}')

            ax.set_xticklabels([])

            if side == 'right':
                ax.invert_xaxis()

        # just RT'ed
        subcategory_key = "RT\'ed"

        fig, axs = plt.subplots(1, 2, figsize=figsize)

        for ax, side in zip(axs, ['left', 'right']):
            counts = edge_distributions[side, subcategory_key]

            if len(counts) == 0:
                continue

            xs = np.arange(min(counts), max(counts) + 1)
            ys = [counts[x] for x in xs]

            ax.plot(xs, ys, '.-', color='black', alpha=0.5)

            pegRNA_name = ti.pegRNA_names_by_side_of_read[side]
            other_pegRNA_name = ti.pegRNA_names_by_side_of_read[twin_prime_layout.other_side[side]]

            # By definition, the end of the PBS on this side's pegRNA 
            # is zero in the coordinate system.
            PBS_end = features[pegRNA_name, 'PBS'].end

            y_start = -0.1

            for feature_name in ['PBS', 'RTT', 'overlap', 'scaffold', 'protospacer']:
                feature = features[pegRNA_name, feature_name]
                
                # On this side's pegRNA, moving back from the PBS end is moving
                # forward in the coordinate system.
                start, end = PBS_end - feature.end - 0.5, PBS_end - feature.start + 0.5
                
                ax.axvspan(start, end, y_start, y_start + ref_bar_height,
                           facecolor=ti.pegRNA_name_to_color[pegRNA_name],
                           clip_on=False,
                          )
                ax.axvspan(start, end, y_start + ref_bar_height, y_start + ref_bar_height + feature_height,
                           facecolor=feature.attribute['color'],
                           alpha=0.75,
                           clip_on=False,
                          )

            ax.set_xlim(*x_lims)
            ax.set_ylim(0)

            ax.set_title(f'{side} {subcategory_key}')

            ax.set_xticklabels([])
                
            if side == 'right':
                ax.invert_xaxis()

        # overlap-extended
        subcategory_key = "RT'ed + overlap-extended"

        fig, axs = plt.subplots(1, 2, figsize=figsize)

        for ax, side in zip(axs, ['left', 'right']):
            counts = edge_distributions[side, subcategory_key]

            if len(counts) == 0:
                continue

            xs = np.arange(min(counts), max(counts) + 1)
            ys = [counts[x] for x in xs]

            ax.plot(xs, ys, '.-', color='black', alpha=0.5)

            pegRNA_name = ti.pegRNA_names_by_side_of_read[side]
            other_pegRNA_name = ti.pegRNA_names_by_side_of_read[twin_prime_layout.other_side[side]]

            # By definition, the end of the PBS on this side's pegRNA 
            # is zero in the coordinate system.
            PBS_end = features[pegRNA_name, 'PBS'].end

            y_start = -0.2

            for feature_name in ['PBS', 'RTT', 'overlap']:
                feature = features[pegRNA_name, feature_name]
                
                # On this side's pegRNA, moving back from the PBS end is moving
                # forward in the coordinate system.
                start, end = PBS_end - feature.end - 0.5, PBS_end - feature.start + 0.5
                
                ax.axvspan(start, end, y_start, y_start + ref_bar_height, facecolor=ti.pegRNA_name_to_color[pegRNA_name], clip_on=False)
                ax.axvspan(start, end, y_start + ref_bar_height, y_start + ref_bar_height + feature_height, facecolor=feature.attribute['color'], alpha=0.75, clip_on=False)
                
            # The left side of the pegRNA overlap in the coordinate system is the 
            # end of the overlap feature on this side's pegRNA.
            overlap_start = PBS_end - features[pegRNA_name, 'overlap'].end

            other_overlap = features[other_pegRNA_name, 'overlap']

            overlap_start_offset = overlap_start - other_overlap.start

            y_start = y_start + ref_bar_height + feature_height + gap_between_refs

            for feature_name in ['PBS', 'RTT', 'overlap']:
                feature = features[other_pegRNA_name, feature_name]
                
                start, end = overlap_start_offset + feature.start - 0.5, overlap_start_offset + feature.end + 0.5
                
                ax.axvspan(start, end, y_start, y_start + ref_bar_height, facecolor=ti.pegRNA_name_to_color[other_pegRNA_name], clip_on=False)
                ax.axvspan(start, end, y_start + ref_bar_height, y_start + ref_bar_height + feature_height, facecolor=feature.attribute['color'], alpha=0.75, clip_on=False)
                
            other_PBS_name = ti.PBS_names_by_side_of_read[twin_prime_layout.other_side[side]]
            other_protospacer_name = pegRNAs.protospacer_name(other_pegRNA_name)
            other_PBS_target = features[ti.target, other_PBS_name]
                
            other_PBS_start_offset = overlap_start_offset + features[other_pegRNA_name, 'PBS'].start

            y_start = y_start + ref_bar_height + feature_height + gap_between_refs

            for feature_name in [other_protospacer_name,
                                 other_PBS_name,
                                 ti.primers_by_side_of_read[twin_prime_layout.other_side[side]].ID,
                                ]:
                feature = features[ti.target, feature_name]
                
                if other_PBS_target.strand == '+':
                    start, end = other_PBS_start_offset + (other_PBS_target.end - feature.end) - 0.5, other_PBS_start_offset + (other_PBS_target.end - feature.start) + 0.5
                else:
                    start, end = other_PBS_start_offset + (feature.start - other_PBS_target.start) - 0.5, other_PBS_start_offset + (feature.end - other_PBS_target.start) + 0.5
                    
                start = max(start, other_PBS_start_offset - 0.5)
                
                if feature_name == other_PBS_name:
                    height = 0.015
                else:
                    height = 0.03
                
                ax.axvspan(start, end,
                           y_start + ref_bar_height,
                           y_start + ref_bar_height + height,
                           facecolor=colors.get(feature_name, feature.attribute['color']),
                           clip_on=False,
                          )

            ax.axvspan(other_PBS_start_offset - 0.5, x_lims[1], y_start, y_start + ref_bar_height, facecolor='C0', clip_on=False)

            ax.set_xlim(*x_lims)
            ax.set_ylim(0)

            ax.set_title(f'{side} {subcategory_key}')

            ax.set_xticklabels([])
                
            if side == 'right':
                ax.invert_xaxis()

        return edge_distributions, joint_distribution

class Bxb1TwinPrimeExperiment(TwinPrimeExperiment):
    @memoized_property
    def categorizer(self):
        return Bxb1_layout.Layout