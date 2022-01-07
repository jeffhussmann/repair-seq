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
        ('deletion + duplication',
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
        ('malformed layout',
            ('low quality',
             'no alignments detected',
            ),
        ), 
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ins_size_to_split_at = 1
        self.categorized = False

    @memoized_property
    def donor_alignments(self):
        ''' pegRNAs are considered donor '''
        d_als = [
            al for al in self.alignments
            if al.reference_name in self.target_info.pegRNA_names
        ]
        
        return d_als

    @memoized_property
    def pegRNA_alignments(self):
        pegRNA_alignments = {
            pegRNA_name: [
                al for al in self.split_target_and_donor_alignments
                if al.reference_name == pegRNA_name
            ]
            for pegRNA_name in self.target_info.pegRNA_names
        }
        
        return pegRNA_alignments

    @memoized_property
    def parsimonious_split_target_and_donor_alignments(self):
        return interval.make_parsimonious(self.split_target_and_donor_alignments)

    def q_to_feature_offset(self, al, feature_name):
        ''' Returns dictionary of {true query position: offset into plus-orientation version of feature '''
        ref_p_to_feature_offset = self.target_info.ref_p_to_feature_offset(al.reference_name, feature_name)
        seq = self.target_info.reference_sequences[al.reference_name]
        
        q_to_feature_offset = {}
        
        for q, read_b, ref_p, ref_b, qual in sam.aligned_tuples(al, seq):
            if ref_p in ref_p_to_feature_offset:
                q_to_feature_offset[q] = ref_p_to_feature_offset[ref_p]
                
        return q_to_feature_offset

    def feature_offset_to_q(self, al, feature_name):
        return utilities.reverse_dictionary(self.q_to_feature_offset(al, feature_name))

    def share_feature(self, first_al, first_feature_name, second_al, second_feature_name):
        '''
        Returns True if any query position is aligned to equivalent positions in first_feature and second_feature
        by first_al and second_al.
        '''
        if first_al is None or second_al is None:
            return False
        
        first_q_to_offsets = self.q_to_feature_offset(first_al, first_feature_name)
        second_q_to_offsets = self.q_to_feature_offset(second_al, second_feature_name)
        
        share_any = any(second_q_to_offsets.get(q) == offset for q, offset in first_q_to_offsets.items())
        
        return share_any

    @memoized_property
    def pegRNA_extension_als(self):
        ti = self.target_info

        pegRNA_extension_als = {}

        for side in ['left', 'right']:
            pegRNA_name = ti.pegRNA_names_by_side_of_read[side]
            pegRNA_seq = ti.reference_sequences[pegRNA_name]
            PBS_name = ti.PBS_names_by_side_of_read[side]

            # Note: can't use parsimonious here.
            pegRNA_als = self.pegRNA_alignments[pegRNA_name]
            target_al = self.target_edge_alignments[side]

            candidate_als = []
            for pegRNA_al in pegRNA_als:
                if self.share_feature(target_al, PBS_name, pegRNA_al, 'PBS'):
                    # If the target edge alignment extends past the candidate pegRNA extension alignment,
                    # ensure that the RTT part of the candidate pegRNA extension alignemnt isn't explained
                    # better by the target edge alignment.
                    if side == 'right':
                        target_al_start = interval.get_covered(target_al).start
                        pegRNA_al_start = interval.get_covered(pegRNA_al).start
                        if target_al_start <= pegRNA_al_start:
                            switch_results = sam.find_best_query_switch_after(target_al, pegRNA_al, ti.target_sequence, pegRNA_seq, min)
                            cropped_pegRNA_extension_al = sam.crop_al_to_query_int(pegRNA_al, switch_results['switch_after'] + 1, len(self.seq))
                        else:
                            cropped_pegRNA_extension_al = pegRNA_al
                    else:
                        # TODO: this code branch hasn't been effectively tested.
                        target_al_end = interval.get_covered(target_al).end
                        pegRNA_al_end = interval.get_covered(pegRNA_al).end
                        if target_al_end >= pegRNA_al_end:
                            switch_results = sam.find_best_query_switch_after(pegRNA_al, target_al, pegRNA_seq, ti.target_sequence, max)
                            cropped_pegRNA_extension_al = sam.crop_al_to_query_int(pegRNA_al, 0, switch_results['switch_after'])
                        else:
                            cropped_pegRNA_extension_al = pegRNA_al

                    RTT_length = sam.feature_overlap_length(cropped_pegRNA_extension_al, self.target_info.features[pegRNA_name, 'RTT'])
                    if RTT_length > 2:
                        candidate_als.append(cropped_pegRNA_extension_al)
                    
            if len(candidate_als) == 1:
                pegRNA_extension_als[side] = candidate_als[0]
            else:
                pegRNA_extension_als[side] = None

        return pegRNA_extension_als

    @memoized_property
    def has_any_pegRNA_extension_al(self):
        return {side for side in ['left', 'right'] if self.pegRNA_extension_als[side] is not None}

    @memoized_property
    def is_intended_edit(self):
        return self.share_feature(self.pegRNA_extension_als['left'], 'overlap', self.pegRNA_extension_als['right'], 'overlap')

    @memoized_property
    def manual_anchors(self):
        ''' Anchors for drawing knock-knock ref-centric diagrams with overlap in pegRNA aligned.
        '''
        ti = self.target_info

        manual_anchors = {}

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
        if present_in_both or present_in_either:
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
                
        # target anchor to center on amplicon
        primer_name = ti.primers_by_side_of_read['left'].attribute['ID']
        ref_p = ti.feature_offset_to_ref_p(ti.target, primer_name)[0]
        q = self.feature_offset_to_q(self.target_edge_alignments['left'], primer_name)[0]

        extra_length = len(ti.amplicon_interval) - self.inferred_amplicon_length
        if ti.sequencing_direction == '-':
            anchor_ref = ref_p - extra_length / 2
        else:
            anchor_ref = ref_p + extra_length / 2

        manual_anchors[ti.target] = (q, anchor_ref)

        return manual_anchors

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
            self.category = 'malformed layout'
            self.subcategory = 'no alignments detected'
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
                        self.category = 'deletion'
                        self.outcome = DeletionOutcome(indel)
                        self.details = str(self.outcome)
                        self.relevant_alignments = [target_alignment]

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
                self.category = 'intended edit'
                self.subcategory = 'replacement'
                self.outcome = Outcome('n/a')
                self.relevant_alignments = self.target_edge_alignments_list + list(self.pegRNA_extension_als.values())
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
                self.relevant_alignments = self.target_edge_alignments_list + list(self.pegRNA_extension_als.values())

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

            elif len(indels) == 1 and indels[0].kind == 'D':
                indel = indels[0]
                if indel in self.target_info.donor_deletions:
                    self.category = 'edit + duplication'
                else:
                    self.category = 'deletion + duplication'

                deletion_outcome = DeletionOutcome(indels[0])
                duplication_outcome = DuplicationOutcome(ref_junctions)
                self.outcome = DeletionPlusDuplicationOutcome(deletion_outcome, duplication_outcome)
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

        else:
            num_Ns = Counter(self.seq)['N']

            if num_Ns > 10:
                self.category = 'malformed layout'
                self.subcategory = 'low quality'
                self.details = 'n/a'

            elif self.Q30_fractions['all'] < 0.5:
                self.category = 'malformed layout'
                self.subcategory = 'low quality'
                self.details = 'n/a'

            elif self.Q30_fractions['second_half'] < 0.5:
                self.category = 'malformed layout'
                self.subcategory = 'low quality'
                self.details = 'n/a'
                
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

    def plot(self, **manual_diagram_kwargs):
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
        )

        diagram_kwargs.update(**manual_diagram_kwargs)

        diagram = knock_knock.visualize.ReadDiagram(self.relevant_alignments,
                                                    ti,
                                                    **diagram_kwargs,
                                                   )

        # If there are any relevant pegRNA alignemnts, draw the pegRNAs.
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
            ref_p_to_xs['left'] = diagram.draw_reference(ti.pegRNA_names_by_side_of_read['left'], 200, ref_ys['left'], True, label_features=False)
            ref_p_to_xs['right'] = diagram.draw_reference(ti.pegRNA_names_by_side_of_read['right'], 200, ref_ys['right'], False)

            diagram.ax.set_xlim(min(old_min_x, ref_p_to_xs['right'](0)),
                                max(old_max_x, ref_p_to_xs['left'](0)),
                               )

            offset_to_ref_ps = ti.feature_offset_to_ref_p(ti.pegRNA_names_by_side_of_read['left'], 'overlap')

            overlap_xs = sorted([ref_p_to_xs['left'](offset_to_ref_ps[0]), ref_p_to_xs['left'](offset_to_ref_ps[max(offset_to_ref_ps)])])
            overlap_xs = knock_knock.visualize.adjust_edges(overlap_xs)

            overlap_color = ti.features[ti.pegRNA_names[0], 'overlap'].attribute['color']
                
            diagram.ax.fill_betweenx([ref_ys['left'], ref_ys['right'] + diagram.feature_line_width],
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

        return diagram