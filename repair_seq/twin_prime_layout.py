from collections import Counter, defaultdict

from hits import interval, sam
from hits import utilities
from hits.utilities import memoized_property

import knock_knock.visualize
from knock_knock.outcome import *

import repair_seq.prime_editing_layout

class Layout(repair_seq.prime_editing_layout.Layout):
    category_order = [
        ('wild type',
            ('clean',
             'mismatches',
             'short indel far from cut',
            ),
        ),
        ('intended edit',
            ('replacement',
             'deletion',
            ),
        ),
        ('unintended annealing of RT\'ed sequence',
            ('left pegRNA',
             'right pegRNA',
             'both pegRNAs',
            ),
        ),
        ('deletion',
            ('clean',
             'mismatches',
             'multiple',
            ),
        ),
        ('duplication',
            ('simple',
             'iterated',
             'complex',
            ),
        ),
        ('insertion',
            ('clean',
             'mismatches',
            ),
        ),
        ('extension from intended annealing',
            ('n/a',
            ),
        ),
        ('genomic insertion',
            ('hg19',
             'bosTau7',
             'e_coli',
            ),
        ),
        ('uncategorized',
            ('uncategorized',
            ),
        ),
        ('nonspecific amplification',
            ('hg19',
             'bosTau7',
             'e_coli',
             'primer dimer',
             'unknown',
            ),
        ),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ins_size_to_split_at = 1
        self.categorized = False

    @memoized_property
    def parsimonious_split_target_and_donor_alignments(self):
        return interval.make_parsimonious(self.split_target_and_donor_alignments)

    @memoized_property
    def has_any_pegRNA_extension_al(self):
        return {side for side in ['left', 'right'] if self.pegRNA_extension_als[side] is not None}

    @memoized_property
    def is_intended_edit(self):
        return self.share_feature(self.pegRNA_extension_als['left'], 'overlap', self.pegRNA_extension_als['right'], 'overlap')

    @memoized_property
    def nonspecific_amplification(self):
        # TODO: port over more sophisticated strategy from kk.layout
        
        covered_from_edges = interval.get_disjoint_covered(list(self.target_edge_alignments.values()))
        
        need_to_cover = self.whole_read - covered_from_edges

        # Try to avoid false positives
        if need_to_cover.total_length <= 20:
            return []

        covering_als = []
        for al in self.supplemental_alignments:
            covered = interval.get_covered(al)
            if len(need_to_cover - covered) == 0:
                covering_als.append(al)
                
        if len(covering_als) == 0:
            covering_als = None
        else:
            covering_als = interval.make_parsimonious(covering_als)

        # If a single-end read doesn't reach the right prime, the amplicon length is unknown.
        if not sam.overlaps_feature(self.target_edge_alignments['right'], self.target_info.primers_by_side_of_read['right']):
            self.trust_inferred_length = False
            
        return covering_als

    def categorize(self):
        self.outcome = None
        self.trust_inferred_length = True

        if len(self.seq) <= self.target_info.combined_primer_length + 10:
            self.category = 'nonspecific amplification'

            if self.non_primer_nts <= 2:
                self.subcategory = 'primer dimer'
            else:
                self.subcategory = 'unknown'

            self.details = 'n/a'
            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.no_alignments_detected:
            self.category = 'uncategorized'
            self.subcategory = 'uncategorized'
            self.details = 'n/a'
            self.outcome = None
            self.trust_inferred_length = False

        elif self.single_read_covering_target_alignment:
            target_alignment = self.single_read_covering_target_alignment
            interesting_indels, uninteresting_indels = self.interesting_and_uninteresting_indels([target_alignment])

            if len(interesting_indels) == 0:
                if self.starts_at_expected_location:
                    self.category = 'wild type'

                    if len(self.non_donor_SNVs) == 0 and len(uninteresting_indels) == 0:
                        self.subcategory = 'clean'
                        self.outcome = Outcome('n/a')

                    elif len(uninteresting_indels) == 1:
                        self.subcategory = 'short indel far from cut'

                        indel = uninteresting_indels[0]
                        if indel.kind == 'D':
                            self.outcome = DeletionOutcome(indel)
                        elif indel.kind == 'I':
                            self.outcome = InsertionOutcome(indel)
                        else:
                            raise ValueError(indel.kind)

                    elif len(uninteresting_indels) > 1:
                        self.category = 'uncategorized'
                        self.subcategory = 'uncategorized'
                        self.outcome = Outcome('n/a')

                    else:
                        self.subcategory = 'mismatches'
                        self.outcome = MismatchOutcome(self.non_donor_SNVs)

                    self.relevant_alignments = [target_alignment]

                else:
                    self.category = 'uncategorized'
                    self.subcategory = 'uncategorized'
                    self.outcome = Outcome('n/a')

                    self.details = str(self.outcome)
                    self.relevant_alignments = [target_alignment]

            elif len(interesting_indels) == 1:
                indel = interesting_indels[0]

                if len(self.non_donor_SNVs) > 0:
                    self.subcategory = 'mismatches'
                else:
                    self.subcategory = 'clean'

                if indel.kind == 'D':
                    if self.non_primer_nts <= 10:
                        self.category = 'nonspecific amplification'
                        self.subcategory = 'primer dimer'
                        self.details = 'n/a'
                        self.relevant_alignments = self.uncategorized_relevant_alignments
                    else:
                        if indel == self.target_info.twin_pegRNA_intended_deletion:
                            self.category = 'intended edit'
                            self.subcategory = 'deletion'
                            self.relevant_alignments = [target_alignment] + self.possible_pegRNA_extension_als_list

                        else:
                            self.category = 'deletion'
                            self.relevant_alignments = [target_alignment]

                        self.outcome = DeletionOutcome(indel)
                        self.details = str(self.outcome)

                elif indel.kind == 'I':
                    self.category = 'insertion'
                    self.outcome = InsertionOutcome(indel)
                    self.details = str(self.outcome)
                    self.relevant_alignments = [target_alignment]

            else: # more than one indel
                if self.non_primer_nts <= 50:
                    self.category = 'nonspecific amplification'
                    self.subcategory = 'unknown'
                    self.details = 'n/a'
                    self.relevant_alignments = self.uncategorized_relevant_alignments

                else:
                    self.category = 'uncategorized'
                    self.subcategory = 'uncategorized'
                    self.details = 'n/a'
                    self.relevant_alignments = self.uncategorized_relevant_alignments

        elif len(self.has_any_pegRNA_extension_al) > 0:
            if self.is_intended_edit:
                if self.target_info.twin_pegRNA_intended_deletion is None:
                    self.category = 'intended edit'
                    self.subcategory = 'replacement'
                    self.outcome = Outcome('n/a')
                    self.relevant_alignments = self.target_edge_alignments_list + self.possible_pegRNA_extension_als_list
                else:
                    self.category = 'uncategorized'
                    self.subcategory = 'uncategorized'
                    self.details = 'n/a'
                    self.relevant_alignments = self.uncategorized_relevant_alignments

            else:
                self.category = 'unintended annealing of RT\'ed sequence'
                if len(self.has_any_pegRNA_extension_al) == 1:
                    side = sorted(self.has_any_pegRNA_extension_al)[0]
                    self.subcategory = f'{side} pegRNA'
                elif len(self.has_any_pegRNA_extension_al) == 2:
                    self.subcategory = f'both pegRNAs'
                else:
                    raise ValueError(len(self.has_any_pegRNA_extension_al))

                self.outcome = Outcome('n/a')
                self.relevant_alignments = self.target_edge_alignments_list + self.possible_pegRNA_extension_als_list

        elif self.long_duplication is not None:
            subcategory, ref_junctions, indels, als_with_donor_SNVs, merged_als = self.long_duplication
            if len(indels) == 0:
                self.outcome = DuplicationOutcome(ref_junctions)

                if als_with_donor_SNVs == 0:
                    self.category = 'duplication'
                else:
                    self.category = 'edit + duplication'

                self.subcategory = subcategory
                self.details = str(self.outcome)
                self.relevant_alignments = merged_als

            else:
                self.category = 'uncategorized'
                self.subcategory = 'uncategorized'
                self.details = 'n/a'
                self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.non_primer_nts <= 50:
            self.category = 'nonspecific amplification'
            self.subcategory = 'unknown'
            self.details = 'n/a'
            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.nonspecific_amplification:
            self.category = 'nonspecific amplification'
            organism, _ = self.target_info.remove_organism_from_alignment(self.nonspecific_amplification[0])
            self.subcategory = organism
            self.details = 'n/a'
            self.relevant_alignments = self.uncategorized_relevant_alignments

        else:
            self.category = 'uncategorized'
            self.subcategory = 'uncategorized'
            self.details = 'n/a'

            self.trust_inferred_length = False
                
            self.relevant_alignments = self.uncategorized_relevant_alignments

        self.relevant_alignments = sam.make_nonredundant(self.relevant_alignments)

        if self.outcome is not None:
            # Translate positions to be relative to a registered anchor
            # on the target sequence.
            self.details = str(self.outcome.perform_anchor_shift(self.target_info.anchor))

        self.categorized = True

        return self.category, self.subcategory, self.details, self.outcome

    @memoized_property
    def manual_anchors(self):
        ''' Anchors for drawing knock-knock ref-centric diagrams with overlap in pegRNA aligned.
        '''
        ti = self.target_info

        manual_anchors = {}

        overlap_length = len(ti.features[ti.pegRNA_names[0], 'overlap'])

        overlap_offset_to_qs = defaultdict(dict)

        for side in ['left', 'right']:
            pegRNA_al = self.pegRNA_extension_als[side]
            
            if pegRNA_al is None:
                continue
            
            pegRNA_name = ti.pegRNA_names_by_side_of_read[side]
            
            overlap_offset_to_qs[side] = self.feature_offset_to_q(pegRNA_al, 'overlap')
            
        present_in_both = sorted(set(overlap_offset_to_qs['left']) & set(overlap_offset_to_qs['right']))
        present_in_either = sorted(set(overlap_offset_to_qs['left']) | set(overlap_offset_to_qs['right']))

        # If there is any offset present in both sides, use it as the anchor.
        # Otherwise, pick any offset present in either side arbitrarily.
        # If there is no such offset, don't make anchors for the pegRNAs.
        # Check overlap length to exclude prime del.
        if overlap_length > 5 and (present_in_both or present_in_either):
            if present_in_both:
                anchor_offset = present_in_both[0]
                qs = [overlap_offset_to_qs[side][anchor_offset] for side in ['left', 'right']] 
                q = int(np.floor(np.mean(qs)))
            elif len(overlap_offset_to_qs['left']) > 0:
                anchor_offset = sorted(overlap_offset_to_qs['left'])[0]
                q = overlap_offset_to_qs['left'][anchor_offset]
            elif len(overlap_offset_to_qs['right']) > 0:
                anchor_offset = sorted(overlap_offset_to_qs['right'])[0]
                q = overlap_offset_to_qs['right'][anchor_offset]

            for side in ['left', 'right']:
                pegRNA_name = ti.pegRNA_names_by_side_of_read[side]
                ref_p = ti.feature_offset_to_ref_p(pegRNA_name, 'overlap')[anchor_offset]
                manual_anchors[pegRNA_name] = (q, ref_p)
                
        ## target anchor to center on amplicon
        #primer_name = ti.primers_by_side_of_read['left'].attribute['ID']
        #ref_p = ti.feature_offset_to_ref_p(ti.target, primer_name)[0]
        #q = self.feature_offset_to_q(self.target_edge_alignments['left'], primer_name).get(0, 0)

        #extra_length = len(ti.amplicon_interval) - self.inferred_amplicon_length
        #if ti.sequencing_direction == '-':
        #    anchor_ref = ref_p - extra_length / 2
        #else:
        #    anchor_ref = ref_p + extra_length / 2

        #manual_anchors[ti.target] = (q, anchor_ref)

        return manual_anchors

    def plot(self, relevant=True, **manual_diagram_kwargs):
        if not self.categorized:
            self.categorize()

        ti = self.target_info

        diagram_kwargs = dict(
            draw_sequence=True,
            flip_target=ti.sequencing_direction == '-',
            split_at_indels=True,
            label_offsets={feature_name: 1 for _, feature_name in ti.PAM_features},
            features_to_show=ti.features_to_show,
            manual_anchors=self.manual_anchors,
            refs_to_draw={ti.target, *ti.pegRNA_names},
            label_overrides={name: 'protospacer' for name in ti.sgRNAs},
            inferred_amplicon_length=self.inferred_amplicon_length,
            center_on_primers=True,
        )

        diagram_kwargs.update(**manual_diagram_kwargs)

        if relevant:
            als_to_plot = self.relevant_alignments
        else:
            als_to_plot = self.uncategorized_relevant_alignments

        diagram = knock_knock.visualize.ReadDiagram(als_to_plot,
                                                    ti,
                                                    **diagram_kwargs,
                                                   )

        # Draw the pegRNAs.
        if any(al.reference_name in ti.pegRNA_names for al in self.relevant_alignments):
            ref_ys = {}
            ref_ys['left'] = diagram.max_y + diagram.target_and_donor_y_gap
            ref_ys['right'] = ref_ys['left'] + 7 * diagram.gap_between_als

            # To ensure that features on pegRNAs that extend far to the right of
            # the read are plotted, temporarily make the x range very wide.
            old_min_x, old_max_x = diagram.min_x, diagram.max_x

            diagram.min_x = -1000
            diagram.max_x = 1000

            ref_p_to_xs = {}

            #if any(al.reference_name == ti.pegRNA_names_by_side_of_read['left'] for al in self.relevant_alignments):
            left_name = ti.pegRNA_names_by_side_of_read['left']
            ref_p_to_xs['left'] = diagram.draw_reference(left_name, ref_ys['left'], True, label_features=False)

            offset_to_ref_ps = ti.feature_offset_to_ref_p(left_name, 'overlap')
            overlap_xs = sorted([ref_p_to_xs['left'](offset_to_ref_ps[0]), ref_p_to_xs['left'](offset_to_ref_ps[max(offset_to_ref_ps)])])

            diagram.max_x = max(old_max_x, ref_p_to_xs['left'](0))

            #else:
            #    diagram.max_x = old_max_x


            #if any(al.reference_name == ti.pegRNA_names_by_side_of_read['right'] for al in self.relevant_alignments):
            right_name = ti.pegRNA_names_by_side_of_read['right']
            ref_p_to_xs['right'] = diagram.draw_reference(right_name, ref_ys['right'], False)

            #offset_to_ref_ps = ti.feature_offset_to_ref_p(ti.pegRNA_names_by_side_of_read['right'], 'overlap')
            #overlap_xs = sorted([ref_p_to_xs['right'](offset_to_ref_ps[0]), ref_p_to_xs['right'](offset_to_ref_ps[max(offset_to_ref_ps)])])

            diagram.min_x = min(old_min_x, ref_p_to_xs['right'](0))
            #else:
            #    diagram.min_x = old_min_x

            diagram.ax.set_xlim(diagram.min_x, diagram.max_x)

            # temporary exclusion of prime del
            if len(offset_to_ref_ps) >= 5:
                overlap_xs = knock_knock.visualize.adjust_edges(overlap_xs)

                overlap_color = ti.features[ti.pegRNA_names[0], 'overlap'].attribute['color']
                    
                diagram.ax.fill_betweenx([ref_ys['left'], ref_ys['right'] + diagram.ref_line_width + diagram.feature_line_width],
                                         [overlap_xs[0], overlap_xs[0]],
                                         [overlap_xs[1], overlap_xs[1]],
                                         color=overlap_color,
                                         alpha=0.5,
                                        )

                text_x = np.mean(overlap_xs)
                text_y = np.mean([ref_ys['left'] + diagram.feature_line_width, ref_ys['right']])
                diagram.ax.annotate('overlap',
                                    xy=(text_x, text_y),
                                    color=overlap_color,
                                    ha='center',
                                    va='center',
                                    size=diagram.font_sizes['feature_label'],
                                    weight='bold',
                                   )

            diagram.update_size()

        return diagram