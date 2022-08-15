import array
from collections import Counter, defaultdict

import numpy as np
import pysam

from hits import interval, sam, utilities, sw, fastq
from hits.utilities import memoized_property

import knock_knock.pegRNAs
from knock_knock.target_info import DegenerateDeletion, DegenerateInsertion, SNV, SNVs
from knock_knock import layout

from knock_knock.outcome import *

class Layout(layout.Categorizer):
    category_order = [
        ('wild type',
            ('clean',
             'short indel far from cut',
             'mismatches',
            ),
        ),
        ('intended edit',
            ('SNV',
             'SNV + mismatches',
             'SNV + short indel far from cut',
             'synthesis errors',
             'deletion',
             'deletion + SNV',
             'deletion + unintended mismatches',
             'insertion',
             'insertion + SNV',
             'insertion + unintended mismatches',
            ),
        ),
        ('unintended rejoining of RT\'ed sequence',
            ('includes scaffold',
             'includes scaffold, no SNV',
             'includes scaffold, with deletion',
             'includes scaffold, no SNV, with deletion',
             'no scaffold',
             'no scaffold, no SNV',
             'no scaffold, with deletion',
             'no scaffold, no SNV, with deletion',
             'doesn\'t include insertion',
            ),
        ),
        ('deletion',
            ('clean',
             'mismatches',
             'multiple',
            ),
        ),
        ('edit + deletion',
            ('edit + deletion',
            ),
        ),
        ('duplication',
            ('simple',
             'iterated',
             'complex',
            ),
        ),
        ('edit + duplication',
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
             'hg38',
             'bosTau7',
             'e_coli',
            ),
        ),
        ('truncation',
            ('clean',
             'mismatches',
            ),
        ),
        ('uncategorized',
            ('uncategorized',
             'indel far from cut',
             'deletion far from cut plus mismatch',
             'deletion plus non-adjacent mismatch',
             'insertion plus one mismatch',
             'pegRNA deletion plus at least one non-pegRNA mismatch',
             'pegRNA SNV with non-pegRNA indel',
             'one indel plus more than one mismatch',
             'multiple indels',
             'multiple indels plus at least one mismatch',
             'multiple indels including pegRNA deletion',
             'multiple indels plus at least one pegRNA SNV',
            ),
        ),
        ('nonspecific amplification',
            ('hg19',
             'hg38',
             'bosTau7',
             'e_coli',
             'primer dimer',
             'short nknown',
             'plasmid',
            ),
        ),
        ('malformed layout',
            ('low quality',
             'no alignments detected',
            ),
        ), 
    ]

    def __init__(self, alignments, target_info, error_corrected=False, mode=None):
        self.alignments = [al for al in alignments if not al.is_unmapped]

        self.target_info = target_info
        
        alignment = alignments[0]

        self.query_name = alignment.query_name

        self.seq = sam.get_original_seq(alignment)
        if self.seq is None:
            self.seq = ''

        self.seq_bytes = self.seq.encode()

        # Note: don't try to make this anything but a python array.
        # pysam will internally try to evaluate it's truth status
        # and fail.
        self.qual = sam.get_original_qual(alignment)

        self.primary_ref_names = set(self.target_info.reference_sequences)

        self.special_alignment = None
        
        self.relevant_alignments = self.alignments

        # 22.01.04: This should probably be 1 - only makes sense to be >1 for DSBs.
        # 22.02.03: doing so breaks current extension from intended annealing logic.
        self.ins_size_to_split_at = 3
        self.del_size_to_split_at = 2

        self.error_corrected = error_corrected
        self.mode = mode

        self.categorized = False

    @classmethod
    def from_read(cls, read, target_info):
        al = pysam.AlignedSegment(target_info.header)
        al.query_sequence = read.seq
        al.query_qualities = read.qual
        al.query_name = read.name
        return cls([al], target_info)
    
    @classmethod
    def from_seq(cls, seq, target_info):
        al = pysam.AlignedSegment(target_info.header)
        al.query_sequence = seq
        al.query_qualities = [41]*len(seq)
        return cls([al], target_info)
        
    @memoized_property
    def target_alignments(self):
        t_als = [
            al for al in self.alignments
            if al.reference_name == self.target_info.target
        ]
        
        return t_als

    @memoized_property
    def all_pegRNA_alignments(self):
        if self.target_info.pegRNA_names is None:
            als = None
        else:
            als = [
                al for al in self.alignments
                if al.reference_name in self.target_info.pegRNA_names
            ]
        
        return als

    @memoized_property
    def pegRNA_alignments(self):
        if self.target_info.pegRNA_names is None:
            pegRNA_alignments = None
        else:
            pegRNA_alignments = {
                pegRNA_name: [
                    al for al in self.split_target_and_pegRNA_alignments
                    if al.reference_name == pegRNA_name
                ]
                for pegRNA_name in self.target_info.pegRNA_names
            }
        
        return pegRNA_alignments

    @memoized_property
    def primary_alignments(self):
        p_als = [
            al for al in self.alignments
            if al.reference_name in self.primary_ref_names
        ]
        
        return p_als

    def pegRNA_alignment_extends_target_alignment(self, pegRNA_al, target_al):
        ti = self.target_info

        pegRNA_name = pegRNA_al.reference_name
        pegRNA_seq = ti.reference_sequences[pegRNA_name]
        PBS_name = knock_knock.pegRNAs.PBS_name(pegRNA_name)
        side = ti.pegRNA_name_to_side_of_read[pegRNA_name]

        after_first_difference_feature = ti.features[pegRNA_name, f'after_first_difference_{pegRNA_name}']

        if self.share_feature(target_al, PBS_name, pegRNA_al, 'PBS'):
            # If the target edge alignment extends past the candidate pegRNA extension alignment,
            # ensure that the RTT part of the candidate pegRNA extension alignment isn't explained
            # better by the target edge alignment.
            target_interval = interval.get_covered(target_al)
            pegRNA_interval = interval.get_covered(pegRNA_al)

            if side == 'right':
                # Include +2 wiggle room here in case pegRNA alignments extends slightly farther by chance.
                if target_interval.start <= pegRNA_interval.start + 2:
                    # Find the farthest right left point from the target al to the pegRNA al.
                    switch_results = sam.find_best_query_switch_after(target_al, pegRNA_al, ti.target_sequence, pegRNA_seq, min)
                    cropped_pegRNA_extension_al = sam.crop_al_to_query_int(pegRNA_al, switch_results['switch_after'] + 1, len(self.seq))
                else:
                    cropped_pegRNA_extension_al = pegRNA_al

            else:
                if target_interval.end >= pegRNA_interval.end:
                    # Find the farthest left switch point from the pegRNA al to the target al.
                    switch_results = sam.find_best_query_switch_after(pegRNA_al, target_al, pegRNA_seq, ti.target_sequence, max)
                    cropped_pegRNA_extension_al = sam.crop_al_to_query_int(pegRNA_al, 0, switch_results['switch_after'])
                else:
                    cropped_pegRNA_extension_al = pegRNA_al

            # If the cropped pegRNA alignment doesn't extend past the point where
            # pegRNA and target diverge, this is not an extension alignment that
            # best explains this part of the read.
            past_first_difference_length = sam.feature_overlap_length(cropped_pegRNA_extension_al, after_first_difference_feature)

            # If the target alignment doesn't explain query outside of the pegRNA alignments,
            # this is not an extension alignment.

            if past_first_difference_length > 0 and target_interval not in pegRNA_interval:
                status = 'definite'
            else:
                status = 'ambiguous'
        else:
            status = None
            cropped_pegRNA_extension_al = None

        return status, cropped_pegRNA_extension_al

    def find_target_alignment_extending_pegRNA_alignment(self, pegRNA_al):
        manually_extended_al = self.extend_target_PBS_alignment(pegRNA_al)
        target_als = self.split_target_alignments + self.target_edge_alignments_list + [manually_extended_al]
        target_als = sam.make_noncontained(target_als)

        relevant_target_al = None

        for target_al in sorted(target_als, key=lambda al: al.query_alignment_length, reverse=True):
            status, cropped_pegRNA_extension_al = self.pegRNA_alignment_extends_target_alignment(pegRNA_al, target_al)
            if status == 'definite':
                relevant_target_al = target_al
                break

        return relevant_target_al

    @memoized_property
    def possible_pegRNA_extension_als(self):
        ''' Identify pegRNA alignments that extend into the RTT and explain
        the observed sequence better than alignment to the target does.
        '''
        ti = self.target_info

        extension_als = {
            'edge_definite': {},
            'edge_ambiguous': {},
            'internal_definite': {},
            'internal_ambiguous': {},
        }

        target_als = self.split_target_alignments + self.target_edge_alignments_list
        target_als = sam.make_nonredundant(target_als)

        for side, pegRNA_name in ti.pegRNA_names_by_side_of_read.items():
            # Note: can't use parsimonious here.
            pegRNA_als = self.pegRNA_alignments[pegRNA_name]

            candidate_als = {
                'edge_definite': [],
                'edge_ambiguous': [],
                'internal_definite': [],
                'internal_ambiguous': [],
            }
            for target_al in target_als:
                for pegRNA_al in pegRNA_als:
                    definite_status, cropped_pegRNA_extension_al = self.pegRNA_alignment_extends_target_alignment(pegRNA_al, target_al)

                    if definite_status is not None:
                        if target_al == self.target_edge_alignments[side]:
                            edge_status = 'edge'
                        else:
                            edge_status = 'internal'

                        key = f'{edge_status}_{definite_status}'
                        
                        candidate_als[key].append(cropped_pegRNA_extension_al)

            for pegRNA_al in pegRNA_als:
                # If a reasonably long pegRNA alignment reaches the end of the read before switching back to target,
                # parsimoniously assume that it eventually does so.
                covered = interval.get_covered(pegRNA_al)

                reaches_right = (side == 'right' and covered.end == self.whole_read.end and len(covered) > 20)
                reaches_left = (side == 'left' and covered.start == 0 and len(covered) > 20)

                # To handle the edge case in which scaffold sequence exists in the target nearby (e.g. in
                # some repair-seq settings), exclude alignments that are entirely covered by
                # a single target alignment.
                within_target_cover = any(covered in interval.get_covered(al) for al in target_als)

                if (reaches_right or reaches_left) and not within_target_cover:
                    # 22.08.08: what status should these get?
                    key = 'edge_definite'

                    candidate_als[key].append(pegRNA_al)

            for status, als in candidate_als.items():
                als = sam.make_nonredundant(als)
                extension_als[status][side] = als

        return extension_als

    @memoized_property
    def pegRNA_extension_als_list(self):
        ''' For plotting/adding to relevant_als '''
        extension_als = []

        for side, als in self.possible_pegRNA_extension_als['edge_definite'].items():
            extension_als.extend(als)

        extension_als = sam.make_nonredundant(extension_als)

        return extension_als

    @memoized_property
    def possible_pegRNA_extension_als_list(self):
        ''' For plotting/adding to relevant_als '''
        extension_als = []

        for key, als_dict in self.possible_pegRNA_extension_als.items():
            for side, als in als_dict.items():
                extension_als.extend(als)

        extension_als = sam.make_nonredundant(extension_als)

        return extension_als

    @memoized_property
    def pegRNA_extension_als(self):
        ''' Unique edge extension als. '''

        extension_als = {
            'left': None,
            'right': None,
        }

        for side, als in self.possible_pegRNA_extension_als['edge_definite'].items():
            if len(als) == 1:
                extension_als[side] = als[0]

        return extension_als

    @memoized_property
    def ambiguous_pegRNA_extension_als(self):
        return self.possible_pegRNA_extension_als['ambiguous']

    @memoized_property
    def flipped_pegRNA_als(self):
        ''' Identify flipped pegRNA alignments that pair the pegRNA protospacer with target protospacer. '''

        ti = self.target_info

        flipped_als = {}

        for side, pegRNA_name in ti.pegRNA_names_by_side_of_read.items():
            flipped_als[side] = []

            # Note: can't use parsimonious here.
            pegRNA_als = self.pegRNA_alignments[pegRNA_name]
            target_al = self.target_edge_alignments[side]

            ps_name = knock_knock.pegRNAs.protospacer_name(pegRNA_name)

            scaffold_feature = ti.features[pegRNA_name, 'scaffold']

            for pegRNA_al in pegRNA_als:
                if self.share_feature(target_al, ps_name, pegRNA_al, 'protospacer'):
                    if sam.feature_overlap_length(pegRNA_al, scaffold_feature) >= 10:
                        flipped_als[side].append(pegRNA_al)
                        
        return flipped_als

    @memoized_property
    def extra_alignments(self):
        ti = self.target_info
        extra_ref_names = {n for n in ti.reference_sequences if n not in [ti.target, *ti.pegRNA_names]}
        als = [al for al in self.alignments if al.reference_name in extra_ref_names]
        return als
    
    @memoized_property
    def supplemental_alignments(self):
        supp_als = [
            al for al in self.alignments
            if al.reference_name not in self.primary_ref_names
        ]

        split_als = []
        for supp_al in supp_als:
            split_als.extend(sam.split_at_large_insertions(supp_al, 2))
        
        few_mismatches = [al for al in split_als if sam.total_edit_distance(al) / al.query_alignment_length < 0.2]

        # Convert relevant supp als to target als, but ignore any that fall largely within the amplicon interval.
        ti = self.target_info
        if ti.reference_name_in_genome_source:
            extended_target_als = []
            other_als = []

            target_interval = interval.Interval(0, len(ti.target_sequence) - 1)

            for al in few_mismatches:
                accounted_for = False

                if al.reference_name == ti.reference_name_in_genome_source:
                    conversion_results = ti.convert_genomic_alignment_to_target_coordinates(al)

                    if conversion_results:
                        converted_interval = interval.Interval(conversion_results['start'], conversion_results['end'])
    
                        if converted_interval in target_interval:
                            al_dict = al.to_dict()

                            al_dict['ref_name'] = ti.target
                            
                            # Note gotcha here: in dictionary form, coordinates need to be 1-based.
                            al_dict['ref_pos'] = str(conversion_results['start'] + 1)

                            converted_al = pysam.AlignedSegment.from_dict(al_dict, ti.header)

                            if converted_al.is_reverse != al.is_reverse:
                                raise NotImplementedError

                            outside_amplicon = interval.get_covered_on_ref(converted_al) - ti.amplicon_interval

                            if outside_amplicon.total_length >= 10:
                                extended_target_als.append(converted_al)
                                accounted_for = True
                            else:
                                # Ignore it.
                                accounted_for = True

                if not accounted_for:
                    other_als.append(al)

            final_als = other_als + extended_target_als
        else:
            final_als = few_mismatches

        return final_als
    
    @memoized_property
    def parsimonious_target_alignments(self):
        ti = self.target_info
        als = interval.make_parsimonious(self.target_gap_covering_alignments)

        if len(als) == 0:
            return als

        # Synthesis errors in primers frequently result in one or more short deletions
        # in the primer and cause alignments to end at one of these deletions.
        # If initial alignments don't reach the read ends, look for more lenient alignments
        # between read edges and primers.
        # An alternative strategy here might be to use sw.extend_repeatedly.

        # If the left edge of the read isn't covered, try to merge a primer alignment to the left-most alignment.
        existing_covered = interval.get_disjoint_covered(als)
        realigned_to_primers = {}

        if existing_covered.start >= 5:
            realigned_to_primers[5] = self.realign_edges_to_primers(5)
            if realigned_to_primers[5] is not None:
                left_most = min(als, key=lambda al: interval.get_covered(al).start)
                others = [al for al in als if al != left_most]
                merged = sam.merge_adjacent_alignments(left_most, realigned_to_primers[5], ti.reference_sequences)
                if merged is None:
                    merged = left_most

                als = others + [merged]

        if self.mode == 'trimmed' and existing_covered.end <= len(self.seq) - 1 - 5:
            realigned_to_primers[3] = self.realign_edges_to_primers(3)
            if realigned_to_primers[3] is not None:
                right_most = max(als, key=lambda al: interval.get_covered(al).end)
                others = [al for al in als if al != right_most]
                merged = sam.merge_adjacent_alignments(right_most, realigned_to_primers[3], ti.reference_sequences)
                if merged is None:
                    merged = right_most

                als = others + [merged]

        # Non-specific amplification of a genomic region that imperfectly matches primers
        # can produce a chimera of the relevant genomic region and primer sequence.
        # Check if more lenient alignments of read edge to primers produces a set of alignments
        # that make up a such a chimera. 

        existing_covered = interval.get_disjoint_covered(als)

        possible_edge_als = []

        if existing_covered.start >= 5:
            possible_edge_als.append(realigned_to_primers[5])

        if self.mode == 'trimmed' and existing_covered.end <= len(self.seq) - 1 - 5:
            possible_edge_als.append(realigned_to_primers[3])

        edge_als = []

        for edge_al in possible_edge_als:
            if edge_al is not None:
                new_covered = interval.get_covered(edge_al) - existing_covered
                # Only add the new alignment if it explains a substantial new amount of the read.
                if new_covered.total_length > 10:
                    edge_als.append(edge_al)

        edge_als_by_side = {'left': [], 'right': []}
        for al in edge_als:
            if sam.get_strand(al) != self.target_info.sequencing_direction:
                continue

            covered = interval.get_covered(al)
            
            if covered.start <= 2:
                edge_als_by_side['left'].append(al)
            
            if covered.end >= len(self.seq) - 1 - 2:
                edge_als_by_side['right'].append(al)

        for edge in ['left', 'right']:
            if len(edge_als_by_side[edge]) > 0:
                best_edge_al = max(edge_als_by_side[edge], key=lambda al: al.query_alignment_length)
                als.append(best_edge_al)

        covered = interval.get_disjoint_covered(als)

        # If the end result of all of these alignment attempts is mergeable alignments,
        # merge them.

        als = sam.merge_any_adjacent_pairs(als, ti.reference_sequences)

        als = [sam.soft_clip_terminal_insertions(al) for al in als]

        return als
    
    @memoized_property
    def split_target_and_pegRNA_alignments(self):
        all_split_als = []

        initial_als = self.target_alignments + self.all_pegRNA_alignments
        if self.perfect_right_edge_alignment is not None:
            initial_als.append(self.perfect_right_edge_alignment)

        for al in initial_als:
            split_als = layout.comprehensively_split_alignment(al,
                                                               self.target_info,
                                                               'illumina',
                                                               self.ins_size_to_split_at,
                                                               self.del_size_to_split_at,
                                                              )

            seq_bytes = self.target_info.reference_sequence_bytes[al.reference_name]

            extended = [sw.extend_alignment(split_al, seq_bytes) for split_al in split_als]

            all_split_als.extend(extended)

        return sam.make_nonredundant(all_split_als)

    @memoized_property
    def split_pegRNA_alignments(self):
        if self.target_info.pegRNA_names is not None:
            valid_names = self.target_info.pegRNA_names

        return [al for al in self.split_target_and_pegRNA_alignments if al.reference_name in valid_names]
    
    @memoized_property
    def split_target_alignments(self):
        return [al for al in self.split_target_and_pegRNA_alignments if al.reference_name == self.target_info.target]

    realign_edges_to_primers = layout.Layout.realign_edges_to_primers
    
    @memoized_property
    def target_edge_alignments(self):
        edge_als = self.get_target_edge_alignments(self.parsimonious_target_alignments)

        return edge_als

    @memoized_property
    def target_edge_alignments_list(self):
        return [al for al in self.target_edge_alignments.values() if al is not None]

    def get_target_edge_alignments(self, alignments, split=True):
        ''' Get target alignments that make it to the read edges. '''
        edge_alignments = {'left': [], 'right':[]}

        if split:
            all_split_als = []
            for al in alignments:
                split_als = layout.comprehensively_split_alignment(al,
                                                                   self.target_info,
                                                                   'illumina',
                                                                   self.ins_size_to_split_at,
                                                                   self.del_size_to_split_at,
                                                                  )
                
                target_seq_bytes = self.target_info.reference_sequences[al.reference_name].encode()
                extended = [sw.extend_alignment(split_al, target_seq_bytes) for split_al in split_als]

                all_split_als.extend(extended)
        else:
            all_split_als = alignments

        for al in all_split_als:
            if sam.get_strand(al) != self.target_info.sequencing_direction:
                continue

            covered = interval.get_covered(al)

            if covered.total_length >= 10:
                if covered.start <= 5 or self.overlaps_primer(al, 'left'):
                    edge_alignments['left'].append(al)
                
                if covered.end >= len(self.seq) - 1 - 5 or self.overlaps_primer(al, 'right'):
                    edge_alignments['right'].append(al)

        for edge in ['left', 'right']:
            if len(edge_alignments[edge]) == 0:
                edge_alignments[edge] = None
            else:
                edge_alignments[edge] = max(edge_alignments[edge], key=lambda al: al.query_alignment_length)

        return edge_alignments

    def overlaps_primer(self, al, side):
        primer = self.target_info.primers_by_side_of_read[side]
        num_overlapping_bases = al.get_overlap(primer.start, primer.end + 1)
        overlaps = num_overlapping_bases > 0
        correct_strand = sam.get_strand(al) == self.target_info.sequencing_direction 

        return al.reference_name == self.target_info.target and correct_strand and overlaps

    @memoized_property
    def whole_read(self):
        return interval.Interval(0, len(self.seq) - 1)

    def whole_read_minus_edges(self, edge_length):
        return interval.Interval(edge_length, len(self.seq) - 1 - edge_length)
    
    @memoized_property
    def single_read_covering_target_alignment_old(self):
        target_als = self.parsimonious_target_alignments
        covering_als = [al for al in target_als if self.alignment_covers_read(al)]
        
        if len(covering_als) == 1:
            return covering_als[0]
        else:
            return None

    @memoized_property
    def single_read_covering_target_alignment(self):
        edge_als = self.target_edge_alignments
        covered = {side: interval.get_covered(al) for side, al in edge_als.items()}

        if covered['right'].start <= covered['left'].end + 1:
            covering_al = sam.merge_adjacent_alignments(edge_als['left'], edge_als['right'], self.target_info.reference_sequences)
            return covering_al
        else:
            return None

    def query_missing_from_alignment(self, al):
        if al is None:
            return None
        else:
            split_als = sam.split_at_large_insertions(al, 5)
            covered = interval.get_disjoint_covered(split_als)
            ignoring_edges = interval.Interval(covered.start, covered.end)

            missing_from = {
                'start': covered.start,
                'end': len(self.seq) - covered.end - 1,
                'middle': (ignoring_edges - covered).total_length,
            }

            return missing_from

    def alignment_covers_read(self, al):
        missing_from = self.query_missing_from_alignment(al)

        # Non-indel-containing alignments can more safely be considered to have truly
        # reached an edge if they make it to a primer since the primer-overlapping part
        # of the alignment is less likely to be noise.
        no_indels = len(self.extract_indels_from_alignments([al])) == 0

        if missing_from is None:
            return False
        else:
            not_too_much = {
                'start': (missing_from['start'] <= 5) or (no_indels and self.overlaps_primer(al, 'left')),
                'end': (missing_from['end'] <= 5) or (no_indels and self.overlaps_primer(al, 'right')),
                'middle': (missing_from['middle'] <= 5),
            }

            starts_at_expected_location = self.overlaps_primer(al, 'left')

            return all(not_too_much.values()) and starts_at_expected_location

    @memoized_property
    def target_reference_edges(self):
        ''' reference positions on target of alignments that make it to the read edges. '''
        edges = {}
        # confusing: 'edge' means 5 or 3, 'side' means left or right here.
        for edge, side in [(5, 'left'), (3, 'right')]:
            edge_al = self.target_edge_alignments[side]
            edges[side] = sam.reference_edges(edge_al)[edge]

        return edges

    @memoized_property
    def starts_at_expected_location(self):
        edge_al = self.target_edge_alignments['left']
        return edge_al is not None and self.overlaps_primer(edge_al, 'left')

    @memoized_property
    def Q30_fractions(self):
        at_least_30 = self.qual >= 30
        fracs = {
            'all': np.mean(at_least_30),
            'second_half': np.mean(at_least_30[len(at_least_30) // 2:]),
        }
        return fracs

    @memoized_property
    def SNVs_summary(self):
        SNPs = self.target_info.donor_SNVs
        if SNPs is None:
            position_to_name = {}
            donor_SNV_locii = {}
        else:
            position_to_name = {SNPs['target'][name]['position']: name for name in SNPs['target']}
            donor_SNV_locii = {name: [] for name in SNPs['target']}

        non_donor_SNVs = []

        for al in self.parsimonious_target_alignments:
            for true_read_i, read_b, ref_i, ref_b, qual in sam.aligned_tuples(al, self.target_info.target_sequence):
                if ref_i in position_to_name:
                    name = position_to_name[ref_i]

                    if SNPs['target'][name]['strand'] == '-':
                        read_b = utilities.reverse_complement(read_b)

                    donor_SNV_locii[name].append((read_b, qual))

                if read_b != '-' and ref_b != '-' and read_b != ref_b:
                    donor_base = self.target_info.simple_donor_SNVs.get((self.target_info.target, ref_i))

                    if donor_base is not None and donor_base == read_b:
                        # Matches donor SNV.
                        pass
                    else:
                        snv = SNV(ref_i, read_b, qual)
                        non_donor_SNVs.append(snv)

        non_donor_SNVs = SNVs(non_donor_SNVs)

        return donor_SNV_locii, non_donor_SNVs

    @memoized_property
    def non_pegRNA_SNVs(self):
        _, non_pegRNA_SNVs = self.SNVs_summary
        return non_pegRNA_SNVs

    def specific_to_pegRNA(self, al):
        ''' Does al contain a pegRNA-specific SNV? '''
        if al is None or al.is_unmapped:
            return False

        ti = self.target_info

        if ti.simple_donor_SNVs is None:
            return False

        ref_name = al.reference_name
        ref_seq = ti.reference_sequences[al.reference_name]

        contains_SNV = False

        for true_read_i, read_b, ref_i, ref_b, qual in sam.aligned_tuples(al, ref_seq):
            # Note: read_b and ref_b are as if the read is the forward strand
            donor_base = ti.simple_donor_SNVs.get((ref_name, ref_i))

            if donor_base is not None and donor_base == read_b:
                contains_SNV = True

        return contains_SNV

    @memoized_property
    def pegRNA_SNV_locii_summary(self):
        SNPs = self.target_info.donor_SNVs
        donor_SNV_locii, _ = self.SNVs_summary
        
        genotype = {}

        has_donor_SNV = False

        for name in sorted(SNPs['target']):
            bs = defaultdict(list)

            for b, q in donor_SNV_locii[name]:
                bs[b].append(q)

            if len(bs) == 0:
                genotype[name] = '-'
            elif len(bs) != 1:
                genotype[name] = 'N'
            else:
                b, qs = list(bs.items())[0]

                if b == SNPs['target'][name]['base']:
                    genotype[name] = '_'
                else:
                    genotype[name] = b
                
                    if b == SNPs['donor'][name]['base']:
                        has_donor_SNV = True

        string_summary = ''.join(genotype[name] for name in sorted(SNPs['target']))

        return has_donor_SNV, string_summary

    @memoized_property
    def has_pegRNA_SNV(self):
        has_pegRNA_SNV, _ = self.pegRNA_SNV_locii_summary
        return has_pegRNA_SNV

    @memoized_property
    def has_any_SNV(self):
        return self.has_pegRNA_SNV or (len(self.non_pegRNA_SNVs) > 0)

    @memoized_property
    def pegRNA_SNV_string(self):
        _, string_summary = self.pegRNA_SNV_locii_summary
        return string_summary

    @memoized_property
    def indels(self):
        return self.extract_indels_from_alignments(self.parsimonious_target_alignments)

    @memoized_property
    def one_base_deletions(self):
        return [indel for indel, near_cut in self.indels if indel.kind == 'D' and indel.length == 1]

    def alignment_scaffold_overlap(self, al):
        ti = self.target_info

        if len(ti.pegRNA_names) != 1:
            raise ValueError(ti.pegRNA_names)

        pegRNA_name = ti.pegRNA_names[0]
        pegRNA_seq = ti.reference_sequences[pegRNA_name]

        scaffold_feature = ti.features[pegRNA_name, 'scaffold']
        cropped = sam.crop_al_to_ref_int(al, scaffold_feature.start, scaffold_feature.end)
        if cropped is None:
            scaffold_overlap = 0
        else:
            scaffold_overlap = cropped.query_alignment_length

            # Try to filter out junk alignments.
            edits = sam.edit_distance_in_query_interval(cropped, ref_seq=pegRNA_seq)
            if edits / scaffold_overlap > 0.2:
                scaffold_overlap = 0

            # Insist on overlapping HA_RT to prevent false positive from protospacer alignment.            
            if not sam.overlaps_feature(al, self.HA_RT, require_same_strand=False):
                scaffold_overlap = 0

        return scaffold_overlap

    @memoized_property
    def max_scaffold_overlap(self):
        return max([self.alignment_scaffold_overlap(al) for al in self.all_pegRNA_alignments], default=0)

    @memoized_property
    def HA_RT(self):
        pegRNA_name = self.target_info.pegRNA_names[0]
        return self.target_info.features[pegRNA_name, f'HA_RT_{pegRNA_name}']

    def deletion_overlaps_HA_RT(self, deletion):
        HA_RT_interval = interval.Interval.from_feature(self.HA_RT)

        deletion_interval = interval.Interval(min(deletion.starts_ats), max(deletion.ends_ats))

        return not interval.are_disjoint(HA_RT_interval, deletion_interval)

    def interesting_and_uninteresting_indels(self, als):
        indels = self.extract_indels_from_alignments(als)

        interesting = []
        uninteresting = []

        for indel, near_cut in indels:
            if near_cut:
                append_to = interesting
            else:
                if indel.kind == 'D' and indel.length == 1:
                    append_to = uninteresting
                else:
                    append_to = interesting

            append_to.append(indel)

        return interesting, uninteresting

    def extract_indels_from_alignments(self, als):
        ti = self.target_info

        around_cut_interval = ti.around_cuts(5)

        primer_intervals = interval.make_disjoint([interval.Interval.from_feature(p) for p in ti.primers.values()])

        indels = []
        for al in als:
            for i, (cigar_op, length) in enumerate(al.cigar):
                if cigar_op == sam.BAM_CDEL:
                    nucs_before = sam.total_reference_nucs(al.cigar[:i])
                    starts_at = al.reference_start + nucs_before
                    ends_at = starts_at + length - 1

                    indel_interval = interval.Interval(starts_at, ends_at)

                    indel = DegenerateDeletion([starts_at], length)

                elif cigar_op == sam.BAM_CINS:
                    ref_nucs_before = sam.total_reference_nucs(al.cigar[:i])
                    starts_after = al.reference_start + ref_nucs_before - 1

                    indel_interval = interval.Interval(starts_after, starts_after)

                    read_nucs_before = sam.total_read_nucs(al.cigar[:i])
                    insertion = al.query_sequence[read_nucs_before:read_nucs_before + length]

                    indel = DegenerateInsertion([starts_after], [insertion])
                    
                else:
                    continue

                near_cut = len(indel_interval & around_cut_interval) > 0
                entirely_in_primer = indel_interval in primer_intervals

                indel = self.target_info.expand_degenerate_indel(indel)
                indels.append((indel, near_cut, entirely_in_primer))

        # Ignore any indels entirely contained in primers.

        indels = [(indel, near_cut) for indel, near_cut, entirely_in_primer in indels if not entirely_in_primer]

        return indels

    @memoized_property
    def indels_string(self):
        reps = [str(indel) for indel in self.indels]
        string = ' '.join(reps)
        return string

    @memoized_property
    def covered_from_target_edges(self):
        als = list(self.target_edge_alignments.values())
        return interval.get_disjoint_covered(als)

    @memoized_property
    def gap_from_target_edges(self):
        edge_als = self.target_edge_alignments

        if edge_als['left'] is None:
            start = 0
        else:
            start = interval.get_covered(edge_als['left']).start
        
        if edge_als['right'] is None:
            end = len(self.seq) - 1
        else:
            end = interval.get_covered(edge_als['right']).end

        ignoring_edges = interval.Interval(start, end)
        gap = ignoring_edges - self.covered_from_target_edges
        if len(gap) > 1:
            raise ValueError
        else:
            gap = gap[0]
        return gap
    
    @memoized_property
    def not_covered_by_target_or_pegRNA(self):
        relevant_als = self.split_target_and_pegRNA_alignments + self.target_edge_alignments_list
        covered = interval.get_disjoint_covered(relevant_als)
        return self.whole_read - covered

    @memoized_property
    def not_covered_by_primary_alignments(self):
        relevant_als = self.primary_alignments
        covered = interval.get_disjoint_covered(relevant_als)
        return self.whole_read - covered

    @memoized_property
    def perfect_gap_als(self):
        all_gap_als = []

        for query_interval in self.not_covered_by_target_or_pegRNA:
            for on in ['target', 'donor']:
                gap_als = self.seed_and_extend(on, query_interval.start, query_interval.end)
                all_gap_als.extend(gap_als)

        return all_gap_als
    
    @memoized_property
    def nonredundant_supplemental_alignments(self):
        nonredundant = []
        
        for al in self.supplemental_alignments:
            covered = interval.get_covered(al)
            novel_covered = covered & self.not_covered_by_target_or_pegRNA & self.not_covered_by_primary_alignments
            if novel_covered:
                nonredundant.append(al)

        return nonredundant

    @memoized_property
    def read(self):
        return fastq.Read(self.query_name, self.seq, fastq.encode_sanger(self.qual))

    def seed_and_extend(self, on, query_start, query_end):
        extender = self.target_info.seed_and_extender[on]
        return extender(self.seq_bytes, query_start, query_end, self.query_name)
    
    @memoized_property
    def valid_intervals_for_edge_alignments(self):
        ti = self.target_info
        forward_primer = ti.primers_by_side_of_target[5]
        reverse_primer = ti.primers_by_side_of_target[3]
        valids = {
            'left': interval.Interval(ti.cut_after + 1, reverse_primer.end),
            #'right': interval.Interval(forward_primer.start, ti.cut_after + 1),
            'right': ti.around_or_between_cuts(5),
        }
        return valids

    @memoized_property
    def perfect_right_edge_alignment(self):
        min_length = 5
        
        edge_al = None

        def is_valid(al):
            long_enough = al.query_alignment_length >= min_length
            overlaps_amplicon_interval = interval.are_overlapping(interval.get_covered_on_ref(al), self.target_info.amplicon_interval)
            return long_enough and overlaps_amplicon_interval

        for length in range(20, 3, -1):
            start = max(0, len(self.seq) - length)
            end = len(self.seq)

            als = self.seed_and_extend('target', start, end)

            valid = [al for al in als if is_valid(al)]
            
        if len(valid) > 0:
            edge_al = max(valid, key=lambda al: al.query_alignment_length)

        return edge_al

    @memoized_property
    def genomic_insertion(self):
        if self.ranked_templated_insertions is None:
            return None
        
        ranked = [details for details in self.ranked_templated_insertions if details['source'] == 'genomic']
        if len(ranked) == 0:
            return None
        else:
            best_explanation = ranked[0]

        return best_explanation

    @memoized_property
    def primer_alignments(self):
        primers = self.target_info.primers_by_side_of_read
        primer_alignments = {}
        for side in ['left', 'right']:
            al = self.target_edge_alignments[side]
            primer_alignments[side] = sam.crop_al_to_feature(al, primers[side])

        return primer_alignments

    @memoized_property
    def covered_by_primers(self):
        return interval.get_disjoint_covered(self.primer_alignments.values())

    @memoized_property
    def not_covered_by_primers(self):
        ''' More complicated than function name suggests. If there are primer
        alignments, returns the interval between but not covered by them, ignoring
        region outside of them, which is likely to be the result of incorrect trimming.
        '''
        if self.covered_by_primers.is_empty:
            return self.whole_read
        elif self.primer_alignments['left'] and not self.primer_alignments['right']:
            return interval.Interval(self.covered_by_primers.end, self.whole_read.end)
        else:
            return interval.Interval(self.covered_by_primers.start, self.covered_by_primers.end) - self.covered_by_primers 
    
    @memoized_property
    def non_primer_nts(self):
        return self.not_covered_by_primers.total_length

    @memoized_property
    def nonspecific_amplification(self):
        ''' Nonspecific amplification if any of following apply:
         - read is empty after adapter trimming
         - read is short after adapter trimming, in which case inferrence of
            nonspecific amplification per se is less clear but
            sequence is unlikely to be informative of any other process
         - read starts with an alignment to the expected primer, but
            this alignment does not extend substantially past the primer.
            Rest of the read is covered by a single alignment to some other
            source that either reaches the end of the read, or reaches an
            an alignment to the other primer (that does not extend 
            substantially past the primer.
        '''
        results = {}

        valid = False

        if len(self.seq) <= self.target_info.combined_primer_length + 10:
            valid = True
            results['covering_als'] = None

        elif self.non_primer_nts <= 10:
            valid = True
            results['covering_als'] = None

        else:
            target_nts_past_primer = {}
            for side in ['left', 'right']:
                target_past_primer = interval.get_covered(self.target_edge_alignments[side]) - interval.get_covered(self.primer_alignments[side])
                target_nts_past_primer[side] = target_past_primer.total_length 

            if self.primer_alignments['left'] is not None:
                if target_nts_past_primer['left'] <= 10 and target_nts_past_primer['right'] <= 10:
                    covering_als = []

                    for al in self.supplemental_alignments + self.split_pegRNA_alignments + self.extra_alignments:
                        covered_by_al = interval.get_covered(al)
                        if len(self.not_covered_by_primers - covered_by_al) == 0:
                            covering_als.append(al)
                
                    if len(covering_als) > 0:
                        valid = True
                        results['covering_als'] = covering_als

        if not valid:
            results = None

        return results

    def register_nonspecific_amplification(self):
        results = self.nonspecific_amplification

        self.category = 'nonspecific amplification'
        self.details = 'n/a'

        if results['covering_als'] is None:
            if self.non_primer_nts <= 2:
                self.subcategory = 'primer dimer'
                self.relevant_alignments = self.target_edge_alignments_list
            else:
                self.subcategory = 'short unknown'
                self.relevant_alignments = sam.make_noncontained(self.uncategorized_relevant_alignments)
        else:
            if self.target_info.pegRNA_names is None:
                pegRNA_names = []
            else:
                pegRNA_names = self.target_info.pegRNA_names

            if any(al.reference_name in pegRNA_names for al in results['covering_als']):
                # amplification off of pegRNA-expressing plasmid
                self.subcategory = 'plasmid'

            elif any(al in self.extra_alignments for al in results['covering_als']):
                # amplification off of other plasmid
                self.subcategory = 'plasmid'
            
            elif any(al.reference_name not in self.primary_ref_names for al in results['covering_als']):
                organisms = {self.target_info.remove_organism_from_alignment(al)[0] for al in results['covering_als']}
                organism = sorted(organisms)[0]
                self.subcategory = organism

            else:
                raise ValueError

            self.relevant_alignments = self.target_edge_alignments_list + results['covering_als']

    def register_genomic_insertion(self):
        details = self.genomic_insertion

        outcome = LongTemplatedInsertionOutcome(details['organism'],
                                                details['chr'],
                                                details['strand'],
                                                details['insertion_reference_bounds'][5],
                                                details['insertion_reference_bounds'][3],
                                                details['insertion_query_bounds'][5],
                                                details['insertion_query_bounds'][3],
                                                details['target_bounds'][5],
                                                details['target_bounds'][3],
                                                details['target_query_bounds'][5],
                                                details['target_query_bounds'][3],
                                                details['MH_lengths']['left'],
                                                details['MH_lengths']['right'],
                                                '',
                                               )

        self.outcome = outcome
        self.category = 'genomic insertion'
        self.subcategory = details['organism']
        self.details = str(outcome)
        self.relevant_alignments = details['full_alignments']
        self.special_alignment = details['cropped_candidate_alignment']

    @memoized_property
    def pegRNA_insertion(self):
        if self.ranked_templated_insertions is None:
            return None
        
        ranked = [details for details in self.ranked_templated_insertions if details['source'] == 'pegRNA']
        if len(ranked) == 0:
            return None
        else:
            best_explanation = ranked[0]
        
        return best_explanation

    def register_pegRNA_insertion(self):
        details = self.pegRNA_insertion

        has_pegRNA_SNV = self.specific_to_pegRNA(details['candidate_alignment'])

        outcome = LongTemplatedInsertionOutcome('pegRNA',
                                                self.target_info.donor,
                                                details['strand'],
                                                details['insertion_reference_bounds'][5],
                                                details['insertion_reference_bounds'][3],
                                                details['insertion_query_bounds'][5],
                                                details['insertion_query_bounds'][3],
                                                details['target_bounds'][5],
                                                details['target_bounds'][3],
                                                details['target_query_bounds'][5],
                                                details['target_query_bounds'][3],
                                                details['MH_lengths']['left'],
                                                details['MH_lengths']['right'],
                                                'PH pegRNA_SNV_summary_string',
                                               )

        relevant_alignments = details['full_alignments']

        sequencing_direction = self.target_info.sequencing_direction
        sgRNA_strand = self.target_info.sgRNA_feature.strand

        if sequencing_direction == '-' and sgRNA_strand == '+':
            pegRNA_end = outcome.left_insertion_ref_bound
            target_start = outcome.left_target_ref_bound

        elif sequencing_direction == '-' and sgRNA_strand == '-':
            pegRNA_end = outcome.right_insertion_ref_bound
            target_start = outcome.right_target_ref_bound

        elif sequencing_direction == '+' and sgRNA_strand == '+':
            pegRNA_end = outcome.right_insertion_ref_bound
            target_start = outcome.right_target_ref_bound

        elif sequencing_direction == '+' and sgRNA_strand == '-':
            pegRNA_end = outcome.left_insertion_ref_bound
            target_start = outcome.left_target_ref_bound

        else:
            raise NotImplementedError

        # 21.02.22 temp fix
        if target_start is None:
            target_start = len(self.target_info.target_sequence)

        # paired_point represents a pegRNA and target position that
        # should be matched with each other in the intended annealing.
        paired_point = self.target_info.offset_to_HA_ref_ps[0]

        # TODO: figure out how to remove use of 'donor' here
        pegRNA_paired_point = paired_point['donor', 'PAM-distal']
        target_paired_point = paired_point['target', 'PAM-distal']
        
        # extension always in negative direction in pegRNA coords
        pegRNA_offset = pegRNA_paired_point - pegRNA_end

        # extension direction in target coords depends on sgRNA strand
        # paired_point is matched and we want offset to be difference from 
        # switching right after matched
        if sgRNA_strand == '+':
            target_offset = target_start - (target_paired_point + 1)
        else:
            # Note: untested if off by 1 here.
            target_offset = (target_paired_point - 1) - target_start

        offset_from_intended_annealing = target_offset - pegRNA_offset

        if self.pegRNA_insertion_matches_intended:
            if details['longest_edge_deletion'] is None or details['longest_edge_deletion'].length <= 2:
                self.category = 'intended edit'
                self.subcategory = 'insertion'
            else:
                self.category = 'edit + deletion'
                self.subcategory = 'edit + deletion'

                HDR_outcome = HDROutcome(self.pegRNA_SNV_string, [])
                deletion_outcome = DeletionOutcome(details['longest_edge_deletion'])
                outcome = HDRPlusDeletionOutcome(HDR_outcome, deletion_outcome)

        # Insist that the extension is a close enough match to the target that
        # the target alignment doesn't get split.
        elif np.abs(offset_from_intended_annealing) <= 1 and self.single_read_covering_target_alignment:
            self.category = 'extension from intended annealing'
            self.subcategory = 'n/a'

        else:
            self.category = 'unintended rejoining of RT\'ed sequence'
            if details['shared_HAs'] == {('left', 'right')}:
                self.subcategory = 'doesn\'t include insertion'
            else:
                if self.alignment_scaffold_overlap(details['cropped_candidate_alignment']) >= 2:
                    self.subcategory = 'includes scaffold'
                else:
                    self.subcategory = 'no scaffold'

                if not has_pegRNA_SNV:
                    self.subcategory += ', no SNV'

                if details['longest_edge_deletion'] is not None:
                    if details['longest_edge_deletion'] != self.target_info.pegRNA_intended_deletion:
                        self.subcategory += ', with deletion'

        self.relevant_alignments = relevant_alignments
        self.special_alignment = details['candidate_alignment']

        self.outcome = outcome
        self.details = str(outcome)
        
    @memoized_property
    def pegRNA_insertion_matches_intended(self):
        return self.templated_insertion_matches_intended(self.pegRNA_insertion)

    def templated_insertion_matches_intended(self, details):
        matches = False

        pegRNA_al = details['candidate_alignment']
        indels = self.extract_indels_from_alignments([pegRNA_al])

        for insertion_feature in self.target_info.pegRNA_programmed_insertions:
            shares_HA = {side: (side, side) in details['shared_HAs'] for side in ['left', 'right']}
            overlaps_feature = sam.overlaps_feature(pegRNA_al, insertion_feature, require_same_strand=False)
            no_indels = len(indels) == 0

            reaches_read_edge = details['insertion_query_bounds'][3] == self.whole_read.end

            if overlaps_feature and no_indels:
                if shares_HA['left']:
                    if shares_HA['right'] or reaches_read_edge:
                        matches = True
        
        return matches

    def shared_HAs(self, pegRNA_al, target_edge_als):
        if self.target_info.homology_arms is None:
            shared_HAs = None
        else:
            shared_HAs = set()
            name_by_side = {side: self.target_info.homology_arms[side]['target'].attribute['ID'] for side in ['left', 'right']}

            for HA_side in ['left', 'right']:
                for target_side in ['left', 'right']:
                    HA_name = name_by_side[HA_side]
                    if self.share_feature(pegRNA_al, HA_name, target_edge_als[target_side], HA_name):
                        shared_HAs.add((HA_side, target_side))

        return shared_HAs

    @memoized_property
    def ranked_templated_insertions(self):
        possible = self.possible_templated_insertions
        valid = [details for details in possible if 'failed' not in details]

        if len(valid) == 0:
            return None

        def priority(details):
            key_order = [
                'total_edits_and_gaps',
                'total_gap_length',
                'edit_distance',
                'gap_before_length',
                'gap_after_length',
                'source',
            ]
            return [details[k] for k in key_order]

        ranked = sorted(valid, key=priority)

        # For performance reasons, only compute some properties on possible insertions that haven't
        # already been ruled out.

        # Assume that edge alignments extending past the expected cut site are just cooincidental
        # sequence match and don't count any such matches as microhomology.
        for details in ranked:
            edge_als_cropped_to_cut = {}
            MH_lengths = {}

            if details['edge_alignments']['left'] is None:
                edge_als_cropped_to_cut['left'] = None
                MH_lengths['left'] = None
            else:
                if details['edge_alignments']['left'].is_reverse:
                    left_of_cut = self.target_info.target_side_intervals[3]
                else:
                    left_of_cut = self.target_info.target_side_intervals[5]

                edge_als_cropped_to_cut['left'] = sam.crop_al_to_ref_int(details['edge_alignments']['left'], left_of_cut.start, left_of_cut.end)

                MH_lengths['left'] = layout.junction_microhomology(self.target_info, edge_als_cropped_to_cut['left'], details['candidate_alignment'])

            if details['edge_alignments']['right'] is None:
                edge_als_cropped_to_cut['right'] = None
                MH_lengths['right'] = None
            else:
                if details['edge_alignments']['right'].is_reverse:
                    right_of_cut = self.target_info.target_side_intervals[5]
                else:
                    right_of_cut = self.target_info.target_side_intervals[3]

                edge_als_cropped_to_cut['right'] = sam.crop_al_to_ref_int(details['edge_alignments']['right'], right_of_cut.start, right_of_cut.end)

                MH_lengths['right'] = layout.junction_microhomology(self.target_info, details['candidate_alignment'], edge_als_cropped_to_cut['right'])

            details['MH_lengths'] = MH_lengths
            details['edge_alignments_cropped_to_cut'] = edge_als_cropped_to_cut

        return ranked

    def evaluate_templated_insertion(self, target_edge_als, candidate_al, source):
        ti = self.target_info

        details = {'source': source}

        candidate_ref_seq = ti.reference_sequences.get(candidate_al.reference_name)
        
        # Find the locations on the query at which switching from edge alignments to the
        # candidate and then back again minimizes the edit distance incurred.
        left_results = sam.find_best_query_switch_after(target_edge_als['left'], candidate_al, ti.target_sequence, candidate_ref_seq, min)
        right_results = sam.find_best_query_switch_after(candidate_al, target_edge_als['right'], candidate_ref_seq, ti.target_sequence, max)

        # For genomic insertions, parsimoniously assign maximal query to candidates that make it all the way to the read edge
        # even if there is a short target alignmnent at the edge.
        if source == 'genomic':
            min_left_results = sam.find_best_query_switch_after(target_edge_als['left'], candidate_al, ti.target_sequence, candidate_ref_seq, min)
            if min_left_results['switch_after'] == -1:
                left_results = min_left_results

            max_right_results = sam.find_best_query_switch_after(candidate_al, target_edge_als['right'], candidate_ref_seq, ti.target_sequence, max)
            if max_right_results['switch_after'] == len(self.seq) - 1:
                right_results = max_right_results

        # TODO: might be valuable to replace max in left_results tie breaker with something that picks the cut site if possible.

        # Crop the alignments at the switch points identified.
        target_bounds = {}
        target_query_bounds = {}

        cropped_left_al = sam.crop_al_to_query_int(target_edge_als['left'], -np.inf, left_results['switch_after'])
        target_bounds[5] = sam.reference_edges(cropped_left_al)[3]
        target_query_bounds[5] = interval.get_covered(cropped_left_al).end

        cropped_right_al = sam.crop_al_to_query_int(target_edge_als['right'], right_results['switch_after'] + 1, np.inf)
        if cropped_right_al is None:
            target_bounds[3] = None
            target_query_bounds[3] = len(self.seq)
        else:
            if cropped_right_al.query_alignment_length >= 8:
                target_bounds[3] = sam.reference_edges(cropped_right_al)[5]
                target_query_bounds[3] = interval.get_covered(cropped_right_al).start
            else:
                target_bounds[3] = None
                target_query_bounds[3] = len(self.seq)

        cropped_candidate_al = sam.crop_al_to_query_int(candidate_al, left_results['switch_after'] + 1, right_results['switch_after'])
        if cropped_candidate_al is None or cropped_candidate_al.is_unmapped:
            details['edge_als'] = target_edge_als
            details['candidate_al'] = candidate_al
            details['switch_afters'] = {'left': left_results['switch_after'], 'right': right_results['switch_after']}
            details['failed'] = 'cropping eliminates insertion'
            return details

        insertion_reference_bounds = sam.reference_edges(cropped_candidate_al)   
        insertion_query_interval = interval.get_covered(cropped_candidate_al)
        insertion_length = len(insertion_query_interval)
            
        left_edits = sam.edit_distance_in_query_interval(cropped_left_al, ref_seq=ti.target_sequence, only_Q30=True)
        right_edits = sam.edit_distance_in_query_interval(cropped_right_al, ref_seq=ti.target_sequence, only_Q30=True)
        middle_edits = sam.edit_distance_in_query_interval(cropped_candidate_al, ref_seq=candidate_ref_seq, only_Q30=True)
        edit_distance = left_edits + middle_edits + right_edits

        gap_before_length = left_results['gap_length']
        gap_after_length = right_results['gap_length']
        total_gap_length = gap_before_length + gap_after_length
        
        has_pegRNA_SNV = {
            'left': self.specific_to_pegRNA(cropped_left_al),
            'right': self.specific_to_pegRNA(cropped_right_al),
        }
        if source == 'pegRNA':
            has_pegRNA_SNV['insertion'] = self.specific_to_pegRNA(candidate_al) # should this be cropped_candidate_al?

        missing_from_blunt = {
            5: None,
            3: None,
        }

        # TODO: there are edge cases where short extra sequence appears at start of read before expected NotI site.
        if target_bounds[5] is not None:
            if target_edge_als['left'].is_reverse:
                missing_from_blunt[5] = target_bounds[5] - (ti.cut_after + 1)
            else:
                missing_from_blunt[5] = ti.cut_after - target_bounds[5]

        if target_bounds[3] is not None:
            missing_from_blunt[3] = (target_bounds[3] - ti.cut_after)
            if target_edge_als['right'].is_reverse:
                missing_from_blunt[3] = ti.cut_after - target_bounds[3]
            else:
                missing_from_blunt[3] = target_bounds[3] - (ti.cut_after + 1)

        longest_edge_deletion = None

        for side in ['left', 'right']:
            if target_edge_als[side] is not None:
                indels = self.extract_indels_from_alignments([target_edge_als[side]])
                for indel, _ in indels:
                    if indel.kind == 'D':
                        if longest_edge_deletion is None or indel.length > longest_edge_deletion.length:
                            longest_edge_deletion = indel

        edit_distance_besides_deletion = edit_distance
        if longest_edge_deletion is not None:
            edit_distance_besides_deletion -= longest_edge_deletion.length

        details.update({
            'source': source,
            'insertion_length': insertion_length,
            'insertion_reference_bounds': insertion_reference_bounds,
            'insertion_query_bounds': {5: insertion_query_interval.start, 3: insertion_query_interval.end},

            'gap_left_query_edge': left_results['switch_after'],
            'gap_right_query_edge': right_results['switch_after'] + 1,

            'gap_before': left_results['gap_interval'],
            'gap_after': right_results['gap_interval'],

            'gap_before_length': gap_before_length,
            'gap_after_length': gap_after_length,
            'total_gap_length': total_gap_length,

            'missing_from_blunt': missing_from_blunt,

            'total_edits_and_gaps': total_gap_length + edit_distance,
            'left_edits': left_edits,
            'right_edits': right_edits,
            'edit_distance': edit_distance,
            'edit_distance_besides_deletion': edit_distance_besides_deletion,
            'candidate_alignment': candidate_al,
            'cropped_candidate_alignment': cropped_candidate_al,
            'target_bounds': target_bounds,
            'target_query_bounds': target_query_bounds,
            'cropped_alignments': [al for al in [cropped_left_al, cropped_candidate_al, cropped_right_al] if al is not None],
            'edge_alignments': target_edge_als,
            'full_alignments': [al for al in [target_edge_als['left'], candidate_al, target_edge_als['right']] if al is not None],

            'longest_edge_deletion': longest_edge_deletion,

            'has_pegRNA_SNV': has_pegRNA_SNV,

            'strand': sam.get_strand(candidate_al),
        })

        if source == 'genomic':
            # The alignment might have been converted to target coordinates.
            if cropped_candidate_al.reference_name == self.target_info.target:
                organism = self.target_info.genome_source
                original_al = cropped_candidate_al
            else:
                organism, original_al = self.target_info.remove_organism_from_alignment(cropped_candidate_al)

            details.update({
                'chr': original_al.reference_name,
                'organism': organism,
                'original_alignment': original_al,
            })

            # Since genomic insertions draw from a much large reference sequence
            # than pegRNA insertions, enforce a more stringent minimum length.

            if insertion_length <= 25:
                details['failed'] = f'insertion length = {insertion_length}'

        failures = []

        if source == 'pegRNA':
            shared_HAs = self.shared_HAs(candidate_al, target_edge_als)
            details['shared_HAs'] = shared_HAs

            # Only want to handle RT extensions here, which should have PAM-distal homology arm usage,
            # unless the read isn't long enough to include this.
            distance_from_end = self.whole_read.end - details['insertion_query_bounds'][3]

            PAM_distal_paired_HA = (ti.PAM_side_to_read_side['PAM-distal'], ti.PAM_side_to_read_side['PAM-distal'])
            insertion_skipping_paired_HA = (ti.PAM_side_to_read_side['PAM-proximal'], ti.PAM_side_to_read_side['PAM-distal'])

            if distance_from_end > 0 and shared_HAs is not None and PAM_distal_paired_HA not in shared_HAs and insertion_skipping_paired_HA not in shared_HAs:
                failures.append('doesn\'t share relevant HA')

        if gap_before_length > 0:
            failures.append(f'gap_before_length = {gap_before_length}')

        if gap_after_length > 0:
            failures.append(f'gap_after_length = {gap_after_length}')

        max_allowable_edit_distance = 5

        # Allow a high edit distance if it is almost entirely explained by a single large deletion.
        if edit_distance_besides_deletion > max_allowable_edit_distance:
            failures.append(f'edit_distance = {edit_distance}')

        if has_pegRNA_SNV['left']:
            failures.append('left alignment has a pegRNA SNV')

        if has_pegRNA_SNV['right']:
            failures.append('right alignment has a pegRNA SNV')

        #if insertion_length < 5:
        #    failures.append(f'insertion length = {insertion_length}')

        edit_distance_over_length = middle_edits / insertion_length
        if edit_distance_over_length >= 0.1:
            failures.append(f'edit distance / length = {edit_distance_over_length}')

        if len(failures) > 0:
            details['failed'] = '; '.join(failures)

        return details

    @memoized_property
    def possible_templated_insertions(self):
        # Make some shorter aliases.
        ti = self.target_info
        # 22.07.14: Removed split=False argument to get_target_edge_alignemnts here.
        edge_als = self.get_target_edge_alignments(self.parsimonious_target_alignments)

        if edge_als['left'] is not None:
            # If a target alignment to the start of the read exists,
            # insist that it be to the sequencing primer. 
            if not sam.overlaps_feature(edge_als['left'], ti.primers_by_side_of_read['left']):
                return [{'failed': 'left edge alignment isn\'t to primer'}]

        candidates = []
        for pegRNA_al in self.split_pegRNA_alignments:
            candidates.append((pegRNA_al, 'pegRNA'))

        for genomic_al in self.nonredundant_supplemental_alignments:
            candidates.append((genomic_al, 'genomic'))

        possible_insertions = []

        for candidate_al, source in candidates:
            details = self.evaluate_templated_insertion(edge_als, candidate_al, source)
            possible_insertions.append(details)

        return possible_insertions

    @memoized_property
    def no_alignments_detected(self):
        return all(al.is_unmapped for al in self.alignments)

    def categorize(self):
        self.outcome = None

        if self.no_alignments_detected:
            self.category = 'malformed layout'
            self.subcategory = 'no alignments detected'
            self.details = 'n/a'
            self.outcome = None

        elif self.nonspecific_amplification:
            self.register_nonspecific_amplification()

        elif self.single_read_covering_target_alignment:
            target_alignment = self.single_read_covering_target_alignment
            interesting_indels, uninteresting_indels = self.interesting_and_uninteresting_indels([target_alignment])

            if len(interesting_indels) == 0:
                if self.has_pegRNA_SNV:
                    self.category = 'intended edit'

                    HDR_outcome = HDROutcome(self.pegRNA_SNV_string, [])
                    self.outcome = HDR_outcome

                    if len(self.non_pegRNA_SNVs) == 0 and len(uninteresting_indels) == 0:
                        self.subcategory = 'SNV'
                    elif len(uninteresting_indels) > 0:
                        if len(uninteresting_indels) == 1:
                            indel = uninteresting_indels[0]
                            if indel.kind == 'D':
                                deletion_outcome = DeletionOutcome(indel)
                                self.outcome = HDRPlusDeletionOutcome(HDR_outcome, deletion_outcome)

                        self.subcategory = 'SNV + short indel far from cut'
                    else:
                        self.subcategory = 'SNV + mismatches'

                    self.details = str(self.outcome)
                    self.relevant_alignments = [target_alignment] + list(self.pegRNA_extension_als.values())

                else:
                    if self.starts_at_expected_location:
                        self.category = 'wild type'

                        if len(self.non_pegRNA_SNVs) == 0 and len(uninteresting_indels) == 0:
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
                            self.outcome = MismatchOutcome(self.non_pegRNA_SNVs)

                    else:
                        self.category = 'uncategorized'
                        self.subcategory = 'uncategorized'
                        self.outcome = Outcome('n/a')

                    self.details = str(self.outcome)
                    self.relevant_alignments = [target_alignment]

            elif self.max_scaffold_overlap >= 2 and self.pegRNA_insertion is not None:
                self.register_pegRNA_insertion()

            elif len(interesting_indels) == 1:
                indel = interesting_indels[0]

                if indel.kind == 'D' and indel == self.target_info.pegRNA_intended_deletion:
                    self.category = 'intended edit'
                    if len(self.non_pegRNA_SNVs) > 0:
                        self.subcategory = 'deletion + unintended mismatches'
                    elif self.has_pegRNA_SNV:
                        self.subcategory = 'deletion + SNV'
                    else:
                        self.subcategory = 'deletion'

                    self.outcome = HDROutcome(self.pegRNA_SNV_string, [indel])
                    self.details = str(self.outcome)

                    self.relevant_alignments = [target_alignment] + self.possible_pegRNA_extension_als_list

                else: # one indel, not a pegRNA deletion
                    if self.has_pegRNA_SNV:
                        if indel.kind == 'D':
                            # If the deletion overlaps with HA_RT on the target, consider this an unintended pegRNA integration.                            
                            #if self.deletion_overlaps_HA_RT(indel) and self.pegRNA_insertion is not None:
                            #    self.register_pegRNA_insertion()
                            self.category = 'edit + deletion'
                            self.subcategory = 'edit + deletion'
                            HDR_outcome = HDROutcome(self.pegRNA_SNV_string, [])
                            deletion_outcome = DeletionOutcome(indel)
                            self.outcome = HDRPlusDeletionOutcome(HDR_outcome, deletion_outcome)
                            self.details = str(self.outcome)
                            self.relevant_alignments = [target_alignment] + self.possible_pegRNA_extension_als_list

                        else:
                            self.category = 'uncategorized'
                            self.subcategory = 'pegRNA SNV with non-pegRNA indel'
                            self.details = 'n/a'
                            self.relevant_alignments = self.uncategorized_relevant_alignments

                    else: # no pegRNA SNVs
                        if len(self.non_pegRNA_SNVs) > 0:
                            self.subcategory = 'mismatches'
                        else:
                            self.subcategory = 'clean'

                        if indel.kind == 'D':
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
                if len(self.indels) == 2:
                    indels = [indel for indel, near_cut in self.indels]
                    if self.target_info.pegRNA_intended_deletion in indels:
                        self.category = 'edit + deletion'
                        HDR_outcome = HDROutcome(self.pegRNA_SNV_string, [self.target_info.pegRNA_intended_deletion])
                        indel = [indel for indel in indels if indel != self.target_info.pegRNA_intended_deletion][0]
                        if indel.kind  == 'D':
                            self.subcategory = 'edit + deletion'
                            deletion_outcome = DeletionOutcome(indel)
                            self.outcome = HDRPlusDeletionOutcome(HDR_outcome, deletion_outcome)
                            self.details = str(self.outcome)
                            self.relevant_alignments = self.parsimonious_target_alignments + self.pegRNA_alignments

                        else:
                            self.category = 'uncategorized'
                            self.subcategory = 'uncategorized'
                            self.details = 'n/a'
                            self.relevant_alignments = self.uncategorized_relevant_alignments

                    elif len([indel for indel in interesting_indels if indel.kind == 'D']) == 2:
                        self.category = 'deletion'
                        self.subcategory = 'multiple'
                        self.outcome = MultipleDeletionOutcome([DeletionOutcome(indel) for indel in interesting_indels])
                        self.details = str(self.outcome)
                        self.relevant_alignments = [target_alignment]
                    else:
                        self.category = 'uncategorized'
                        self.subcategory = 'uncategorized'
                        self.details = 'n/a'
                        self.relevant_alignments = self.uncategorized_relevant_alignments

                else:
                    self.category = 'uncategorized'
                    self.subcategory = 'uncategorized'
                    self.details = 'n/a'
                    self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.duplication_covers_whole_read:
            subcategory, ref_junctions, indels, als_with_pegRNA_SNVs, merged_als = self.duplication
            self.outcome = DuplicationOutcome(ref_junctions)

            if als_with_pegRNA_SNVs == 0:
                self.category = 'duplication'
            else:
                self.category = 'edit + duplication'

            self.subcategory = subcategory
            self.details = str(self.outcome)
            self.relevant_alignments = self.possible_pegRNA_extension_als_list + merged_als

        elif self.pegRNA_insertion is not None:
            self.register_pegRNA_insertion()

        elif self.duplication is not None:
            subcategory, ref_junctions, indels, als_with_pegRNA_SNVs, merged_als = self.duplication
            if len(indels) == 0:
                self.outcome = DuplicationOutcome(ref_junctions)

                if als_with_pegRNA_SNVs == 0:
                    self.category = 'duplication'
                else:
                    self.category = 'edit + duplication'

                self.subcategory = subcategory
                self.details = str(self.outcome)
                self.relevant_alignments = self.possible_pegRNA_extension_als_list + merged_als

            elif len(indels) == 1 and indels[0].kind == 'D':
                indel = indels[0]
                if indel == self.target_info.pegRNA_intended_deletion:
                    self.category = 'edit + duplication'
                else:
                    self.category = 'deletion + duplication'

                deletion_outcome = DeletionOutcome(indels[0])
                duplication_outcome = DuplicationOutcome(ref_junctions)
                self.outcome = DeletionPlusDuplicationOutcome(deletion_outcome, duplication_outcome)
                self.subcategory = subcategory
                self.details = str(self.outcome)
                self.relevant_alignments = self.possible_pegRNA_extension_als_list + merged_als

            else:
                self.category = 'uncategorized'
                self.subcategory = 'uncategorized'
                self.details = 'n/a'
                self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.duplication_plus_insertion is not None:
            self.category = 'edit + duplication'
            self.subcategory = 'simple'
            self.details = 'n/a'
            self.relevant_alignments = self.duplication_plus_insertion

        elif self.genomic_insertion is not None:
            self.register_genomic_insertion()

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

            self.relevant_alignments = self.uncategorized_relevant_alignments

        self.relevant_alignments = sam.make_nonredundant(self.relevant_alignments)

        if self.outcome is not None:
            # Translate positions to be relative to a registered anchor
            # on the target sequence.
            self.details = str(self.outcome.perform_anchor_shift(self.target_info.anchor))

        self.categorized = True

        return self.category, self.subcategory, self.details, self.outcome

    @memoized_property
    def alignments_to_search_for_gaps(self):
        return self.split_target_and_pegRNA_alignments

    @memoized_property
    def gap_covering_alignments(self):
        ti = self.target_info

        initial_als = self.alignments_to_search_for_gaps

        initial_uncovered = self.whole_read_minus_edges(5) - interval.get_disjoint_covered(initial_als)
        
        gap_covers = []
        
        target_interval = ti.amplicon_interval
        
        for gap in initial_uncovered:
            if gap.total_length == 1:
                continue

            start = max(0, gap.start - 5)
            end = min(len(self.seq) - 1, gap.end + 5)
            extended_gap = interval.Interval(start, end)

            als = sw.align_read(self.read,
                                [(ti.target, ti.target_sequence),
                                ],
                                4,
                                ti.header,
                                N_matches=False,
                                max_alignments_per_target=5,
                                read_interval=extended_gap,
                                ref_intervals={ti.target: target_interval},
                                mismatch_penalty=-2,
                               )

            als = [sw.extend_alignment(al, ti.reference_sequence_bytes[ti.target]) for al in als]
            
            gap_covers.extend(als)

            if ti.pegRNA_names is not None:
                for pegRNA_name in ti.pegRNA_names:
                    als = sw.align_read(self.read,
                                        [(pegRNA_name, ti.reference_sequences[pegRNA_name]),
                                        ],
                                        4,
                                        ti.header,
                                        N_matches=False,
                                        max_alignments_per_target=5,
                                        read_interval=extended_gap,
                                        mismatch_penalty=-2,
                                    )

                    als = [sw.extend_alignment(al, ti.reference_sequence_bytes[pegRNA_name]) for al in als]
                    
                    gap_covers.extend(als)

        all_als = initial_als + gap_covers

        return sam.make_nonredundant(all_als)

    @memoized_property
    def target_gap_covering_alignments(self):
        return [al for al in self.gap_covering_alignments if al.reference_name == self.target_info.target]

    @memoized_property
    def pegRNA_gap_covering_alignments(self):
        pegRNA_names = self.target_info.pegRNA_names
        return [al for al in self.gap_covering_alignments if pegRNA_names is not None and al.reference_name in pegRNA_names]

    @memoized_property
    def duplications_from_each_read_edge(self):
        ti = self.target_info
        target_als = interval.make_parsimonious(self.target_gap_covering_alignments)
        # Order target als by position on the query from left to right.
        target_als = sorted(target_als, key=interval.get_covered)

        correct_strand_als = [al for al in target_als if sam.get_strand(al) == ti.sequencing_direction]

        # Need deletions to be merged.
        merged_als = sam.merge_any_adjacent_pairs(correct_strand_als, ti.reference_sequences)
        
        intervals = [interval.get_covered(al) for al in merged_als]
        
        if len(merged_als) > 0 and self.overlaps_primer(merged_als[0], 'left'):
            no_gaps_through_index = 0
            
            for i in range(1, len(intervals)):
                cumulative_from_left = interval.make_disjoint(intervals[:i + 1])
                
                # If there are no gaps so far
                if len(cumulative_from_left.intervals) == 1:
                    no_gaps_through_index = i
                else:
                    break
                    
            from_left_edge = merged_als[:no_gaps_through_index + 1]
        else:
            from_left_edge = None
            
        if len(merged_als) > 0 and \
           (self.overlaps_primer(merged_als[len(intervals) - 1], 'right') or
            (intervals[-1].end >= self.whole_read.end - 1 and len(intervals[-1]) >= 20)
           ):
            no_gaps_through_index = len(intervals) - 1
            
            for i in range(len(intervals) - 2, 0, -1):
                cumulative_from_right = interval.make_disjoint(intervals[i:])
                
                # If there are no gaps so far
                if len(cumulative_from_right.intervals) == 1:
                    no_gaps_through_index = i
                else:
                    break
                    
            from_right_edge = merged_als[no_gaps_through_index:]
        else:
            from_right_edge = None
        
        return from_left_edge, from_right_edge

    @memoized_property
    def duplication_plus_insertion(self):
        from_left_edge, from_right_edge = self.duplications_from_each_read_edge

        explains_whole_read = False

        if from_left_edge is not None:
            left_edge = from_left_edge[-1]
        else:
            left_edge = None
            
        if from_right_edge is not None:
            right_edge = from_right_edge[0]
        else:
            right_edge = None

        edge_als = {
            'left': left_edge,
            'right': right_edge,
        }

        if 'right' in self.pegRNA_extension_als and self.pegRNA_extension_als['right'] is not None:
            pegRNA_al = self.pegRNA_extension_als['right']
            details = self.evaluate_templated_insertion(edge_als, pegRNA_al, 'pegRNA')

            if 'failed' not in details:
                if self.templated_insertion_matches_intended(details):
                    explains_whole_read = True

        if explains_whole_read:
            alignments = [pegRNA_al]

            if from_left_edge is not None:
                alignments.extend(from_left_edge)

            if from_right_edge is not None:
                alignments.extend(from_right_edge)
        else:
            alignments = None

        return alignments

    @memoized_property
    def duplication(self):
        ''' (duplication, simple)   - a single junction
            (duplication, iterated) - multiple uses of the same junction
            (duplication, complex)  - multiple junctions that are not exactly the same
        '''
        ti = self.target_info
        target_als = interval.make_parsimonious(self.target_gap_covering_alignments)
        # Order target als by position on the query from left to right.
        target_als = sorted(target_als, key=interval.get_covered)

        correct_strand_als = [al for al in target_als if sam.get_strand(al) == ti.sequencing_direction]

        merged_als = sam.merge_any_adjacent_pairs(correct_strand_als, ti.reference_sequences)

        covereds = []
        for al in merged_als:
            covered = interval.get_covered(al)
            if covered.total_length >= 20:
                if self.overlaps_primer(al, 'right'):
                    covered.end = self.whole_read.end
                if self.overlaps_primer(al, 'left'):
                    covered.start = 0
            covereds.append(covered)
    
        covered = interval.make_disjoint(covereds)

        uncovered = self.whole_read_minus_edges(2) - covered
        
        if len(merged_als) == 1 or uncovered.total_length > 0:
            return None
        
        ref_junctions = []

        indels = []

        als_with_pegRNA_SNVs = sum(self.specific_to_pegRNA(al) for al in merged_als)

        indels = [indel for indel, _ in self.extract_indels_from_alignments(merged_als)]

        for junction_i, (left_al, right_al) in enumerate(zip(merged_als, merged_als[1:])):
            switch_results = sam.find_best_query_switch_after(left_al, right_al, ti.target_sequence, ti.target_sequence, max)

            lefts = tuple(sam.closest_ref_position(q, left_al) for q in switch_results['best_switch_points'])
            rights = tuple(sam.closest_ref_position(q + 1, right_al) for q in switch_results['best_switch_points'])

            ref_junction = (lefts, rights)
            ref_junctions.append(ref_junction)

        if len(ref_junctions) == 1:
            subcategory = 'simple'
        elif len(set(ref_junctions)) == 1:
            # There are multiple junctions but they are all identical.
            subcategory = 'iterated'
        else:
            subcategory = 'complex'

        return subcategory, ref_junctions, indels, als_with_pegRNA_SNVs, merged_als

    @memoized_property
    def duplication_covers_whole_read(self):
        if self.duplication is None:
            return False
        else:
            _, _, indels, _, merged_als = self.duplication
            not_covered = self.whole_read - interval.get_disjoint_covered(merged_als)
            return (not_covered.total_length == 0) and (len(indels) == 0)

    def extend_target_PBS_alignment(self, pegRNA_al):
        ''' Takes takes a pegRNA alignment and generates
        the longest perfectly extended target alignment that pairs with the PBS.
        Motivation: if a potential transition from a pegRNA alignment back to 
        genomic (target) sequence at a PBS occurs close to the end of a read,
        initial alignment generation may fail to produce a relevant target alignment
        that potentially extends a few nts past the PBS, especially if there is a
        mismatch in the PBS.
        ''' 

        ti = self.target_info

        pegRNA_name = pegRNA_al.reference_name

        target_PBS = ti.features[ti.target, knock_knock.pegRNAs.PBS_name(pegRNA_name)]

        pegRNA_PBS = ti.features[pegRNA_name, 'PBS']

        pegRNA_PBS_al = sam.crop_al_to_feature(pegRNA_al, pegRNA_PBS)

        if pegRNA_PBS_al is None or pegRNA_PBS_al.is_unmapped or sam.contains_indel(pegRNA_PBS_al):
            extended_al = None
        else:
            # Reminder: feature_offset is relative to "plus-orientation version of feature",
            # so min of feature_offsets will be the right-most for a plus-orientation feature
            # but max will be right-most for a minus-orientation feature.
            pegRNA_feature_offset_to_q = self.feature_offset_to_q(pegRNA_al, 'PBS')

            if target_PBS.strand == '+':
                start_offset = min(pegRNA_feature_offset_to_q)
            else:
                start_offset = max(pegRNA_feature_offset_to_q)

            target_PBS_start = ti.feature_offset_to_ref_p(target_PBS.seqname, target_PBS.ID)[start_offset]

            query_start = min(pegRNA_feature_offset_to_q.values())
            query_end = max(pegRNA_feature_offset_to_q.values())

            al = pysam.AlignedSegment(ti.header)

            al.query_sequence = self.seq
            al.query_qualities = self.qual

            soft_clip_before = query_start
            soft_clip_after = len(self.seq) - 1 - query_end

            al.cigar = [
                (sam.BAM_CSOFT_CLIP, soft_clip_before),
                (sam.BAM_CMATCH, pegRNA_PBS_al.query_alignment_length),
                (sam.BAM_CSOFT_CLIP, soft_clip_after),
            ]

            if ti.sequencing_direction == '-':
                # Need to extract query_qualities before overwriting query_sequence.
                flipped_query_qualities = al.query_qualities[::-1]
                al.query_sequence = utilities.reverse_complement(al.query_sequence)
                al.query_qualities = flipped_query_qualities
                al.is_reverse = True
                al.cigar = al.cigar[::-1]

            al.reference_name = ti.target
            al.query_name = self.read.name
            al.next_reference_id = -1

            al.reference_start = target_PBS_start

            extended_al = sw.extend_alignment(al, ti.reference_sequence_bytes[ti.target])

        return extended_al

    @memoized_property
    def longest_polyG(self):
        locations = utilities.homopolymer_lengths(self.seq, 'G')

        if locations:
            max_length = max(length for p, length in locations)
        else:
            max_length = 0

        return max_length

    def is_pegRNA_protospacer_alignment(self, al):
        ''' Returns True if al aligns almost entirely to a protospacer region of a pegRNA
        typically for the purpose of deciding whether to plot it.
        '''
        ti = self.target_info
        
        if ti.pegRNA_names is None:
            return False
        
        if al.reference_name not in ti.pegRNA_names:
            return False
        
        PS_feature = ti.features[al.reference_name, 'protospacer']
        outside_protospacer = sam.crop_al_to_ref_int(al, PS_feature.end + 1, np.inf)

        return (outside_protospacer is None or outside_protospacer.query_alignment_length <= 3)

    @memoized_property
    def uncategorized_relevant_alignments(self):
        als = self.gap_covering_alignments + interval.make_parsimonious(self.nonredundant_supplemental_alignments + self.extra_alignments)
        als = [al for al in als if not self.is_pegRNA_protospacer_alignment(al)]

        if self.target_info.pegRNA_names is None:
            pegRNA_names = []
        else:
            pegRNA_names = self.target_info.pegRNA_names

        pegRNA_als = [al for al in als if al.reference_name in pegRNA_names]
        target_als = [al for al in als if al.reference_name == self.target_info.target]
        other_als = [al for al in als if al.reference_name not in pegRNA_names + [self.target_info.target]]

        pegRNA_als = sam.make_noncontained(pegRNA_als)

        for al in pegRNA_als:
            # If it is already an extension al, the corresponding target al must already exist.
            if al not in self.pegRNA_extension_als_list:
                extended_al = self.extend_target_PBS_alignment(al)
                if extended_al is not None:
                    target_als.append(extended_al)

        target_als = sam.make_noncontained(target_als)

        als = pegRNA_als + target_als + other_als

        als = sam.make_noncontained(als, max_length=10)

        return als

    @property
    def inferred_amplicon_length(self):
        ''' Infer the length of the amplicon including the portion
        of primers that is present in the genome. To prevent synthesis
        errors in primers from shifting this slightly, identify the
        distance in the query between the end of the left primer and
        the start of the right primer, then add the expected length of
        both primers to this. If the sequencing read is single-end
        and doesn't reach the right primer but ends in an alignment
        to the target, parsimoniously assume that this alignment continues
        on through the primer to infer length.
        ''' 

        if self.seq  == '':
            return 0
        elif (self.whole_read - self.covered_by_primers).total_length == 0:
            return len(self.seq)
        elif len(self.seq) <= 50:
            return len(self.seq)

        left_al = self.target_edge_alignments['left']
        right_al = self.target_edge_alignments['right']

        left_primer = self.target_info.primers_by_side_of_read['left']
        right_primer = self.target_info.primers_by_side_of_read['right']

        left_offset_to_q = self.feature_offset_to_q(left_al, left_primer.ID)
        right_offset_to_q = self.feature_offset_to_q(right_al, right_primer.ID)

        # Only trust the inferred length if there are non-spurious target alignments
        # to both edges.
        def is_nonspurious(al):
            min_nonspurious_length = 15
            return al is not None and al.query_alignment_length >= min_nonspurious_length

        if is_nonspurious(left_al) and is_nonspurious(right_al):
            # Calculate query distance between inner primer edges.
            if len(left_offset_to_q) > 0:
                left_inner_edge_offset = max(left_offset_to_q)
                left_inner_edge_q = left_offset_to_q[left_inner_edge_offset]
            else:
                left_inner_edge_q = 0

            if len(right_offset_to_q) > 0:
                right_inner_edge_offset = max(right_offset_to_q)
                right_inner_edge_q = right_offset_to_q[right_inner_edge_offset]
            else:
                right_inner_edge_q = sam.query_interval(right_al)[1]

            # *_inner_edge_q is last position in the primer, so shift each by one to 
            # have boundaries of region between them.
            length_seen_between_primers = (right_inner_edge_q - 1) - (left_inner_edge_q + 1) + 1

            right_al_edge_in_target = sam.reference_edges(right_al)[3]

            # Calculated inferred unseen length.
            if self.target_info.sequencing_direction == '+':
                distance_to_right_primer = right_primer.start - right_al_edge_in_target
            else:
                distance_to_right_primer = right_al_edge_in_target - right_primer.end

            # right_al might extend past the primer start, so only care about positive values.
            inferred_extra_at_end = max(distance_to_right_primer, 0)

            # Combine seen with inferred unseen and expected primer legnths.
            inferred_length = length_seen_between_primers + inferred_extra_at_end + len(left_primer) + len(right_primer)

        else:
            inferred_length = -1

        return inferred_length

    @memoized_property
    def manual_anchors(self):
        ''' Anchors for drawing knock-knock ref-centric diagrams with overlap in pegRNA aligned.
        '''
        ti = self.target_info

        manual_anchors = {}

        extension_als = [al for al in self.pegRNA_extension_als.values() if al is not None]

        if len(extension_als) > 0:
            pegRNA_name = ti.pegRNA_names[0]
            extension_al = extension_als[0]

            PBS_offset_to_qs = self.feature_offset_to_q(extension_al, 'PBS')
                
            if PBS_offset_to_qs:
                anchor_offset = sorted(PBS_offset_to_qs)[0]
                q = PBS_offset_to_qs[anchor_offset]

                ref_p = ti.feature_offset_to_ref_p(pegRNA_name, 'PBS')[anchor_offset]

                manual_anchors[pegRNA_name] = (q, ref_p)
                
        return manual_anchors

    def plot(self, relevant=True, **manual_diagram_kwargs):
        if not self.categorized:
            self.categorize()

        ti = self.target_info

        pegRNA_name = ti.pegRNA_names[0]

        PBS_name = knock_knock.pegRNAs.PBS_name(pegRNA_name)
        PBS_strand = ti.features[ti.target, PBS_name].strand

        flip_target = ti.sequencing_direction == '-'

        if (flip_target and PBS_strand == '-') or (not flip_target and PBS_strand == '+'):
            flip_pegRNA = True
        else:
            flip_pegRNA = False

        label_overrides = {}
        label_offsets = {}
        feature_heights = {}

        for _, feature_name in self.target_info.PAM_features:
            label_offsets[feature_name] = 1

        for deletion in self.target_info.pegRNA_programmed_deletions:
            label_offsets[deletion.ID] = 1
            label_overrides[deletion.ID] = 'intended deletion'
            feature_heights[deletion.ID] = 0.5

        for name in ti.sgRNAs:
            label_overrides[name] = 'protospacer'

        label_overrides.update({feature_name: None for _, feature_name in ti.PAM_features})

        for pegRNA in self.target_info.pegRNA_names:
            label_offsets[f'insertion_{pegRNA}'] = 1

        features_to_show = {*ti.features_to_show}
        features_to_show.update({(ti.target, name) for name in ti.sgRNAs})
        features_to_show.update(sorted(ti.PAM_features))

        diagram_kwargs = dict(
            draw_sequence=True,
            flip_target=flip_target,
            split_at_indels=True,
            features_to_show=features_to_show,
            manual_anchors=self.manual_anchors,
            refs_to_draw={ti.target, *ti.pegRNA_names},
            label_offsets=label_offsets,
            label_overrides=label_overrides,
            inferred_amplicon_length=self.inferred_amplicon_length,
            center_on_primers=True,
            highlight_SNPs=True,
            feature_heights=feature_heights,
        )

        for k, v in diagram_kwargs.items():
            manual_diagram_kwargs.setdefault(k, v)

        if relevant:
            als_to_plot = self.relevant_alignments
        else:
            als_to_plot = self.uncategorized_relevant_alignments

        diagram = knock_knock.visualize.ReadDiagram(als_to_plot,
                                                    ti,
                                                    **manual_diagram_kwargs,
                                                   )

        # Note that diagram.alignments may be different than als_to_plot
        # due to application of parsimony.

        # Draw the pegRNA.
        if any(al.reference_name in ti.pegRNA_names for al in diagram.alignments):
            ref_y = diagram.max_y + diagram.target_and_donor_y_gap

            # To ensure that features on pegRNAs that extend far to the right of
            # the read are plotted, temporarily make the x range very wide.
            old_min_x, old_max_x = diagram.min_x, diagram.max_x

            diagram.min_x = -1000
            diagram.max_x = 1000

            ref_p_to_xs = diagram.draw_reference(pegRNA_name, ref_y, flip_pegRNA)

            pegRNA_seq = ti.reference_sequences[pegRNA_name]
            pegRNA_min_x, pegRNA_max_x = sorted([ref_p_to_xs(0), ref_p_to_xs(len(pegRNA_seq) - 1)])

            diagram.max_x = max(old_max_x, pegRNA_max_x)

            diagram.min_x = min(old_min_x, pegRNA_min_x)

            diagram.ax.set_xlim(diagram.min_x, diagram.max_x)

            diagram.update_size()

        return diagram