import copy

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pysam

from hits import interval, sam, utilities, sw, fastq
from hits.utilities import memoized_property

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
        ('edit + indel',
            ('edit + deletion',
             'edit + insertion',
            ),
        ),
        ('edit + duplication',
            ('simple',
             'iterated',
             'complex',
            ),
        ),
        ('deletion',
            ('clean',
             'mismatches',
             'multiple',
            ),
        ),
        ('deletion + adjacent mismatch',
            ('deletion + adjacent mismatch',
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
        ('SD-MMEJ',
            ('loop-out',
             'snap-back',
             'multi-step',
            ),
        ),
        ('duplication',
            ('simple',
             'iterated',
             'complex',
            ),
        ),
        ('extension from intended annealing',
            ('n/a',
            ),
        ),
        ('unintended annealing of RT\'ed sequence',
            ('includes scaffold',
             'includes scaffold, no SNV',
             'includes scaffold, with deletion',
             'includes scaffold, no SNV, with deletion',
             'no scaffold',
             'no scaffold, no SNV',
             'no scaffold, with deletion',
             'no scaffold, no SNV, with deletion',
            ),
        ),
        ('genomic insertion',
            ('hg19',
             'bosTau7',
             'e_coli',
            ),
        ),
        ('truncation',
            ('clean',
             'mismatches',
            ),
        ),
        ('phiX',
            ('phiX',
            ),
        ),
        ('uncategorized',
            ('uncategorized',
             'indel far from cut',
             'deletion far from cut plus mismatch',
             'deletion plus non-adjacent mismatch',
             'insertion plus one mismatch',
             'donor deletion plus at least one non-donor mismatch',
             'donor SNV with non-donor indel',
             'one indel plus more than one mismatch',
             'multiple indels',
             'multiple indels plus at least one mismatch',
             'multiple indels including donor deletion',
             'multiple indels plus at least one donor SNV',
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

    def __init__(self, alignments, target_info, error_corrected=False, mode=None):
        self.alignments = [al for al in alignments if not al.is_unmapped]
        self.target_info = target_info
        
        alignment = alignments[0]
        self.query_name = alignment.query_name
        self.seq = sam.get_original_seq(alignment)
        if self.seq is None:
            self.seq = ''

        self.seq_bytes = self.seq.encode()
        self.qual = np.array(sam.get_original_qual(alignment))

        self.primary_ref_names = set(self.target_info.reference_sequences)

        self.required_sw = False

        self.special_alignment = None
        
        self.relevant_alignments = self.alignments

        self.ins_size_to_split_at = 3
        self.del_size_to_split_at = 2

        self.error_corrected = error_corrected
        self.mode = mode

        self.trust_inferred_length = True

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
    def donor_alignments(self):
        d_als = [
            al for al in self.alignments
            if al.reference_name == self.target_info.donor
        ]
        
        return d_als
    
    @memoized_property
    def extra_alignments(self):
        ti = self.target_info
        extra_ref_names = {n for n in ti.reference_sequences if n not in [ti.target, ti.donor]}
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
        
        return few_mismatches
    
    @memoized_property
    def phiX_alignments(self):
        als = [
            al for al in self.alignments
            if al.reference_name == 'phiX'
        ]
        
        return als

    def covers_whole_read(self, al):
        if al is None:
            return False

        covered = interval.get_covered(al)

        return len(self.whole_read - covered) == 0

    @memoized_property
    def parsimonious_target_alignments(self):
        ti = self.target_info
        #als = interval.make_parsimonious(self.split_target_alignments)
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
            if sam.get_strand(al) != self.expected_target_strand:
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

        # One-sided sequencing reads of outcomes that represent complex rearrangments may end
        # a short way into a new segment that is too short to produce an initial alignment.
        # To catch some such cases, look for perfect alignments between uncovered read edges
        # and perfect alignment to relevant sequences.

        covered = interval.get_disjoint_covered(als)

        if len(self.seq) - 1 not in covered:
            right_most = max(als, key=lambda al: interval.get_covered(al).end)
            other = [al for al in als if al != right_most]

            perfect_edge_als = self.perfect_edge_alignments
            merged = sam.merge_adjacent_alignments(right_most, perfect_edge_als['right'], ti.reference_sequences)
            if merged is None:
                merged = right_most

            als = other + [merged]
        
        if 0 not in covered:
            left_most = min(als, key=lambda al: interval.get_covered(al).start)
            other = [al for al in als if al != left_most]

            perfect_edge_als = self.perfect_edge_alignments
            merged = sam.merge_adjacent_alignments(perfect_edge_als['left'], left_most, ti.reference_sequences)
            if merged is None:
                merged = left_most

            als = other + [merged]

        # If the end result of all of these alignment attempts is mergeable alignments,
        # merge them.

        als = sam.merge_any_adjacent_pairs(als, ti.reference_sequences)

        als = [sam.soft_clip_terminal_insertions(al) for al in als]

        return als
    
    @memoized_property
    def split_target_and_donor_alignments(self):
        all_split_als = []
        for al in self.target_alignments + self.donor_alignments:
            split_als = layout.comprehensively_split_alignment(al,
                                                               self.target_info,
                                                               'illumina',
                                                               self.ins_size_to_split_at,
                                                               self.del_size_to_split_at,
                                                              )

            if al.reference_name == self.target_info.target:
                seq_bytes = self.target_info.target_sequence_bytes
            else:
                seq_bytes = self.target_info.donor_sequence_bytes

            extended = [sw.extend_alignment(split_al, seq_bytes) for split_al in split_als]

            all_split_als.extend(extended)

        return sam.make_nonredundant(all_split_als)

    @memoized_property
    def split_donor_alignments(self):
        return [al for al in self.split_target_and_donor_alignments if al.reference_name == self.target_info.donor]
    
    @memoized_property
    def split_target_alignments(self):
        return [al for al in self.split_target_and_donor_alignments if al.reference_name == self.target_info.target]

    realign_edges_to_primers = layout.Layout.realign_edges_to_primers
    
    @memoized_property
    def target_edge_alignments(self):
        edge_als = self.get_target_edge_alignments(self.parsimonious_target_alignments)

        return edge_als

    @memoized_property
    def target_edge_alignments_list(self):
        return [al for al in self.target_edge_alignments.values() if al is not None]

    @memoized_property
    def expected_target_strand(self):
        ti = self.target_info
        return ti.features[ti.target, 'sequencing_start'].strand
    
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
    def single_read_covering_target_alignment(self):
        target_als = self.parsimonious_target_alignments
        covering_als = [al for al in target_als if self.alignment_covers_read(al)]
        
        if len(covering_als) == 1:
            return covering_als[0]
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
                'start': missing_from['start'] <= 5 or (no_indels and self.overlaps_primer(al, 'left')),
                'end': (missing_from['end'] <= 5) or (no_indels and self.overlaps_primer(al, 'right')),
                'middle': missing_from['middle'] <= 5,
            }

            return all(not_too_much.values())

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

        other_locii = []

        for al in self.parsimonious_target_alignments:
            for true_read_i, read_b, ref_i, ref_b, qual in sam.aligned_tuples(al, self.target_info.target_sequence):
                if ref_i in position_to_name:
                    name = position_to_name[ref_i]

                    if SNPs['target'][name]['strand'] == '-':
                        read_b = utilities.reverse_complement(read_b)

                    donor_SNV_locii[name].append((read_b, qual))

                else:
                    if read_b != '-' and ref_b != '-' and read_b != ref_b:
                        snv = SNV(ref_i, read_b, qual)
                        other_locii.append(snv)

        other_locii = SNVs(other_locii)

        return donor_SNV_locii, other_locii

    @memoized_property
    def non_donor_SNVs(self):
        _, other_locii = self.SNVs_summary
        return other_locii

    def donor_al_SNV_summary(self, donor_al):
        ti = self.target_info
        SNPs = self.target_info.donor_SNVs

        if donor_al is None or donor_al.is_unmapped or donor_al.reference_name != ti.donor or ti.simple_donor_SNVs is None:
            return {}

        ref_seq = ti.reference_sequences[donor_al.reference_name]
        
        position_to_name = {SNPs['donor'][name]['position']: name for name in SNPs['donor']}

        SNV_summary = {name: '-' for name in ti.donor_SNVs['donor']}

        for true_read_i, read_b, ref_i, ref_b, qual in sam.aligned_tuples(donor_al, ref_seq):
            # Note: read_b and ref_b are as if the read is the forward strand
            name = position_to_name.get(ref_i)
            if name is None:
                continue

            target_base = SNPs['target'][name]['base']

            # not confident that the logic is right here
            if SNPs['target'][name]['strand'] != SNPs['donor'][name]['strand']:
                read_b = utilities.reverse_complement(read_b)

            if read_b == target_base:
                SNV_summary[name] = '_'
            else:
                SNV_summary[name] = read_b

        string_summary = ''.join(SNV_summary[name] for name in sorted(SNPs['target']))
                
        return string_summary

    def specific_to_donor(self, al):
        ''' Does al contain a donor SNV? '''
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
    def donor_SNV_locii_summary(self):
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
    def has_donor_SNV(self):
        has_donor_SNV, _ = self.donor_SNV_locii_summary
        return has_donor_SNV

    @memoized_property
    def has_any_SNV(self):
        return self.has_donor_SNV or (len(self.non_donor_SNVs) > 0)

    @memoized_property
    def donor_SNV_string(self):
        _, string_summary = self.donor_SNV_locii_summary
        return string_summary

    @memoized_property
    def indels(self):
        return self.extract_indels_from_alignments(self.parsimonious_target_alignments)

    @memoized_property
    def one_base_deletions(self):
        return [indel for indel, near_cut in self.indels if indel.kind == 'D' and indel.length == 1]

    def alignment_scaffold_overlap(self, al):
        ti = self.target_info
        scaffold_feature = ti.features[ti.donor, 'scaffold']
        cropped = sam.crop_al_to_ref_int(al, scaffold_feature.start, scaffold_feature.end)
        if cropped is None:
            scaffold_overlap = 0
        else:
            scaffold_overlap = cropped.query_alignment_length

            # Try to filter out junk alignments.
            edits = sam.edit_distance_in_query_interval(cropped, ref_seq=ti.donor_sequence)
            if edits / scaffold_overlap > 0.2:
                scaffold_overlap = 0

            # Insist on overlapping HA_RT to prevent false positive from protospacer alignment.            
            if not sam.overlaps_feature(al, self.HA_RT, require_same_strand=False):
                scaffold_overlap = 0

        return scaffold_overlap

    @memoized_property
    def max_scaffold_overlap(self):
        return max([self.alignment_scaffold_overlap(al) for al in self.donor_alignments], default=0)

    @memoized_property
    def HA_RT(self):
        ti = self.target_info
        HA_RT_name = [feature_name for seq_name, feature_name in ti.features if seq_name == ti.donor and feature_name.startswith('HA_RT')][0]
        HA_RT = ti.homology_arms[HA_RT_name]['donor']
        return HA_RT

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
    def not_covered_by_target_or_donor(self):
        covered = interval.get_disjoint_covered(self.split_target_and_donor_alignments)
        return self.whole_read - covered

    @memoized_property
    def perfect_gap_als(self):
        all_gap_als = []

        for query_interval in self.not_covered_by_target_or_donor:
            for on in ['target', 'donor']:
                gap_als = self.seed_and_extend(on, query_interval.start, query_interval.end)
                all_gap_als.extend(gap_als)

        return all_gap_als
    
    @memoized_property
    def nonredundant_supplemental_alignments(self):
        nonredundant = []
        
        for al in self.supplemental_alignments:
            covered = interval.get_covered(al)
            novel_covered = covered & self.not_covered_by_target_or_donor
            if novel_covered:
                nonredundant.append(al)

        return nonredundant

    @memoized_property
    def original_header(self):
        return self.alignments[0].header

    @memoized_property
    def read(self):
        return fastq.Read(self.query_name, self.seq, fastq.encode_sanger(self.qual))

    @memoized_property
    def sw_alignments(self):
        self.required_sw = True

        ti = self.target_info

        targets = [
            (ti.target, ti.target_sequence),
        ]

        if ti.donor is not None:
            targets.append((ti.donor, ti.donor_sequence))

        stringent_als = sw.align_read(self.read, targets, 5, ti.header,
                                      max_alignments_per_target=10,
                                      mismatch_penalty=-8,
                                      indel_penalty=-60,
                                      min_score_ratio=0,
                                      both_directions=True,
                                      N_matches=False,
                                     )

        no_Ns = [al for al in stringent_als if 'N' not in al.get_tag('MD')]

        return no_Ns
    
    def sw_interval_to_donor(self, query_start, query_end):
        ti = self.target_info
        
        seq = self.seq[query_start:query_end + 1]
        read = fastq.Read('read', seq, fastq.encode_sanger([41]*len(seq)))
        
        als = sw.align_read(read, [(ti.donor, ti.donor_sequence)], 5, ti.header,
                            alignment_type='whole_query',
                            min_score_ratio=0.5,
                            indel_penalty=None,
                            deletion_penalty=-2,
                            mismatch_penalty=-2,
                            insertion_penalty=-10,
                           )
        if len(als) == 0:
            return None
        
        al = als[0]
        
        before_cigar = [(sam.BAM_CSOFT_CLIP, query_start)]
        after_cigar = [(sam.BAM_CSOFT_CLIP, len(self.seq) - 1 - query_end)]
        if al.is_reverse:
            cigar = after_cigar + al.cigar + before_cigar
            al.query_sequence = utilities.reverse_complement(self.seq)
        else:
            cigar = before_cigar + al.cigar + after_cigar
            al.query_sequence = self.seq

        al.cigar = cigar

        al = sw.extend_alignment(al, ti.donor_sequence_bytes)
        al.query_qualities = [41] * len(self.seq)
        
        return al

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
    def perfect_edge_alignments_and_gap(self):
        # Set up keys to prioritize alignments.
        # Prioritize by (correct strand and side of cut, length (longer better), distance from cut (closer better)) 
        
        def sort_key(al, side):
            length = al.query_alignment_length

            valid_int = self.valid_intervals_for_edge_alignments[side]
            valid_strand = self.target_info.sequencing_direction

            correct_strand = sam.get_strand(al) == valid_strand

            cropped = sam.crop_al_to_ref_int(al, valid_int.start, valid_int.end)
            if cropped is None or cropped.is_unmapped:
                correct_side = 0
                inverse_distance = 0
            else:
                correct_side = 1

                if side == 'left':
                    if valid_strand == '+':
                        edge = cropped.reference_end - 1
                    else:
                        edge = cropped.reference_start
                else:
                    if valid_strand == '+':
                        edge = cropped.reference_start
                    else:
                        edge = cropped.reference_end - 1

                inverse_distance = 1 / (abs(edge - self.target_info.cut_after) + 0.1)

            return correct_strand & correct_side, length, inverse_distance

        def is_valid(al, side):
            correct_strand_and_side, length, inverse_distance = sort_key(al, side)
            return correct_strand_and_side
        
        best_edge_als = {'left': None, 'right': None}

        # Insist that the alignments be to the right side and strand, even if longer ones
        # to the wrong side or strand exist.
        for side in ['left', 'right']:
            for length in range(20, 3, -1):
                if side == 'left':
                    start = 0
                    end = length
                else:
                    start = max(0, len(self.seq) - length)
                    end = len(self.seq)

                als = self.seed_and_extend('target', start, end)

                valid = [al for al in als if is_valid(al, side)]
                if len(valid) > 0:
                    break
                
            if len(valid) > 0:
                key = lambda al: sort_key(al, side)
                best_edge_als[side] = max(valid, key=key)

        covered_from_edges = interval.get_disjoint_covered(best_edge_als.values())
        uncovered = self.whole_read - covered_from_edges
        
        if uncovered.total_length == 0:
            gap_interval = None
        elif len(uncovered.intervals) > 1:
            # This shouldn't be possible since seeds start at each edge
            raise ValueError('disjoint gap', uncovered)
        else:
            gap_interval = uncovered.intervals[0]
        
        return best_edge_als, gap_interval

    @memoized_property
    def perfect_edge_alignments(self):
        edge_als, gap = self.perfect_edge_alignments_and_gap
        return edge_als

    @memoized_property
    def gap_between_perfect_edge_als(self):
        edge_als, gap = self.perfect_edge_alignments_and_gap
        return gap

    @memoized_property
    def perfect_edge_alignment_reference_edges(self):
        edge_als = self.perfect_edge_alignments

        # TODO: this isn't general for a primer upstream of cut
        left_edge = sam.reference_edges(edge_als['left'])[3]
        right_edge = sam.reference_edges(edge_als['right'])[5]

        return left_edge, right_edge

    def reference_distances_from_perfect_edge_alignments(self, al):
        al_edges = sam.reference_edges(al)
        left_edge, right_edge = self.perfect_edge_alignment_reference_edges 
        return abs(left_edge - al_edges[5]), abs(right_edge - al_edges[3])

    def perfect_gap_covering_alignments(self, required_MH_start, required_MH_end, only_close=False):
        def close_enough(al):
            if not only_close:
                return True
            else:
                return min(*self.reference_distances_from_perfect_edge_alignments(al)) < 100

        longest_edge_als, gap_interval = self.perfect_edge_alignments_and_gap
        gap_query_start = gap_interval.start - required_MH_start
        # Note: interval end is the last base, but seed_and_extend wants one past
        gap_query_end = gap_interval.end + 1 + required_MH_end
        gap_covering_als = self.seed_and_extend('target', gap_query_start, gap_query_end)
        gap_covering_als = [al for al in gap_covering_als if close_enough(al)]
        
        return gap_covering_als
    
    def partial_gap_perfect_alignments(self, required_MH_start, required_MH_end, on='target', only_close=True):
        def close_enough(al):
            if not only_close:
                return True
            else:
                return min(*self.reference_distances_from_perfect_edge_alignments(al)) < 100

        edge_als, gap_interval = self.perfect_edge_alignments_and_gap
        if gap_interval is None:
            return [], []

        # Note: interval end is the last base, but seed_and_extend wants one past
        start = gap_interval.start - required_MH_start
        end = gap_interval.end + 1 + required_MH_end

        from_start_gap_als = []
        while (end > start) and not from_start_gap_als:
            end -= 1
            from_start_gap_als = self.seed_and_extend(on, start, end)
            from_start_gap_als = [al for al in from_start_gap_als if close_enough(al)]
            
        start = gap_interval.start - required_MH_start
        end = gap_interval.end + 1 + required_MH_end
        from_end_gap_als = []
        while (end > start) and not from_end_gap_als:
            start += 1
            from_end_gap_als = self.seed_and_extend(on, start, end)
            from_end_gap_als = [al for al in from_end_gap_als if close_enough(al)]

        return from_start_gap_als, from_end_gap_als

    @memoized_property
    def multi_step_SD_MMEJ_gap_cover(self):
        partial_als = {}
        partial_als['start'], partial_als['end'] = self.partial_gap_perfect_alignments(2, 2)
        def is_valid(al):
            close_enough = min(self.reference_distances_from_perfect_edge_alignments(al)) < 50
            return close_enough and not self.target_info.overlaps_cut(al)

        valid_als = {side: [al for al in partial_als[side] if is_valid(al)] for side in ('start', 'end')}
        intervals = {side: [interval.get_covered(al) for al in valid_als[side]] for side in ('start', 'end')}

        part_of_cover = {'start': set(), 'end': set()}

        valid_cover_found = False
        for s, start_interval in enumerate(intervals['start']):
            for e, end_interval in enumerate(intervals['end']):
                if len((start_interval & end_interval)) >= 2:
                    valid_cover_found = True
                    part_of_cover['start'].add(s)
                    part_of_cover['end'].add(e)

        if valid_cover_found:
            final_als = {side: [valid_als[side][i] for i in part_of_cover[side]] for side in ('start', 'end')}
            return final_als
        else:
            return None

    @memoized_property
    def SD_MMEJ(self):
        details = {}

        best_edge_als, gap_interval = self.perfect_edge_alignments_and_gap
        overlaps_cut = self.target_info.overlaps_cut

        if best_edge_als['left'] is None or best_edge_als['right'] is None:
            details['failed'] = 'missing edge alignment'
            return details

        details['edge alignments'] = best_edge_als
        details['alignments'] = list(best_edge_als.values())
        details['all alignments'] = list(best_edge_als.values())
        
        if best_edge_als['left'] == best_edge_als['right']:
            details['failed'] = 'perfect wild type'
            return details
        elif sam.get_strand(best_edge_als['left']) != sam.get_strand(best_edge_als['right']):
            details['failed'] = 'edges align to different strands'
            return details
        else:
            edge_als_strand = sam.get_strand(best_edge_als['left'])
        
        details['left edge'], details['right edge'] = self.perfect_edge_alignment_reference_edges

        # Require resection on both sides of the cut.
        for edge in ['left', 'right']:
            if overlaps_cut(best_edge_als[edge]):
                details['failed'] = f'{edge} edge alignment extends over cut'
                return details

        if gap_interval is None:
            details['failed'] = 'no gap' 
            return details

        details['gap length'] = len(gap_interval)

        # Insist on at least 2 nt of MH on each side.
        gap_covering_als = self.perfect_gap_covering_alignments(2, 2)

        min_distance = np.inf
        closest_gap_covering = None

        for al in gap_covering_als:
            left_distance, right_distance = self.reference_distances_from_perfect_edge_alignments(al)
            distance = min(left_distance, right_distance)
            if distance < 100:
                details['all alignments'].append(al)

            # A valid gap covering alignment must lie entirely on one side of the cut site in the target.
            if distance < min_distance and not overlaps_cut(al):
                min_distance = distance
                closest_gap_covering = al

        # Empirically, the existence of any gap alignments that cover cut appears to be from overhang duplication, not SD-MMEJ.
        # but not comfortable excluding these yet
        #if any(overlaps_cut(al) for al in gap_covering_als):
        #    details['failed'] = 'gap alignment overlaps cut'
        #    return details
        
        if min_distance <= 50:
            details['gap alignment'] = closest_gap_covering
            gap_edges = sam.reference_edges(closest_gap_covering)
            details['gap edges'] = {'left': gap_edges[5], 'right': gap_edges[3]}

            if closest_gap_covering is not None:
                gap_covering_strand = sam.get_strand(closest_gap_covering)
                if gap_covering_strand == edge_als_strand:
                    details['kind'] = 'loop-out'
                else:
                    details['kind'] = 'snap-back'

            details['alignments'].append(closest_gap_covering)

            gap_covered = interval.get_covered(closest_gap_covering)
            edge_covered = {side: interval.get_covered(best_edge_als[side]) for side in ['left', 'right']}
            homology_lengths = {side: len(gap_covered & edge_covered[side]) for side in ['left', 'right']}

            details['homology lengths'] = homology_lengths
            
        else:
            # Try to cover with multi-step.
            multi_step_als = self.multi_step_SD_MMEJ_gap_cover
            if multi_step_als is not None:
                for side in ['start', 'end']:
                    details['alignments'].extend(multi_step_als[side])
                details['kind'] = 'multi-step'
                details['gap edges'] = {'left': 'PH', 'right': 'PH'}
                details['homology lengths'] = {'left': 'PH', 'right': 'PH'}
            else:
                details['failed'] = 'no valid alignments cover gap' 
                return details

        return details

    @memoized_property
    def is_valid_SD_MMEJ(self):
        return 'failed' not in self.SD_MMEJ

    @memoized_property
    def realigned_target_alignments(self):
        return [al for al in self.sw_alignments if al.reference_name == self.target_info.target]
    
    @memoized_property
    def realigned_donor_alignments(self):
        return [al for al in self.sw_alignments if al.reference_name == self.target_info.donor]
    
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
    def non_primer_nts(self):
        primers = self.target_info.primers_by_side_of_read
        left_al = self.target_edge_alignments['left']
        if left_al is None:
            return len(self.seq)
        left_primer_interval = interval.Interval.from_feature(primers['left'])
        left_al_cropped_to_primer = sam.crop_al_to_ref_int(left_al, left_primer_interval.start, left_primer_interval.end)

        if left_al_cropped_to_primer is None:
            return len(self.seq)

        right_al = self.target_edge_alignments['right']
        if right_al is None:
            return len(self.seq)
        right_primer_interval = interval.Interval.from_feature(primers['right'])
        right_al_cropped_to_primer = sam.crop_al_to_ref_int(right_al, right_primer_interval.start, right_primer_interval.end)

        if right_al_cropped_to_primer is None:
            return len(self.seq)

        covered_by_primers = interval.get_disjoint_covered([left_al_cropped_to_primer, right_al_cropped_to_primer])
        not_covered_between_primers = interval.Interval(covered_by_primers.start, covered_by_primers.end) - covered_by_primers 

        return not_covered_between_primers.total_length

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

        # Check for non-specific amplification.
        if outcome.left_target_ref_bound is None:
            left_target_past_primer = 0
        else:
            left_primer = self.target_info.primers_by_side_of_read['left']
            if self.target_info.sequencing_direction == '-':
                left_target_past_primer = left_primer.start - outcome.left_target_ref_bound
            else:
                left_target_past_primer = outcome.left_target_ref_bound - left_primer.end
            
        if outcome.right_target_ref_bound is None:
            right_target_past_primer = 0
        else:
            right_primer = self.target_info.primers_by_side_of_read['right']
            if self.target_info.sequencing_direction == '-':
                right_target_past_primer = outcome.right_target_ref_bound - right_primer.end
            else:
                right_target_past_primer = right_primer.start - outcome.right_target_ref_bound

        if left_target_past_primer < 10 and right_target_past_primer < 10 and details['total_gap_length'] < 5:
            self.category = 'nonspecific amplification'
            self.subcategory = outcome.source

        else:
            self.category = 'genomic insertion'
            self.subcategory = details['organism']

        self.details = str(outcome)
        self.relevant_alignments = details['full_alignments']
        self.special_alignment = details['cropped_candidate_alignment']
        self.trust_inferred_length = False

    @memoized_property
    def donor_insertion(self):
        if self.ranked_templated_insertions is None:
            return None
        
        ranked = [details for details in self.ranked_templated_insertions if details['source'] == 'donor']
        if len(ranked) == 0:
            return None
        else:
            best_explanation = ranked[0]
        
        return best_explanation

    def register_donor_insertion(self):
        details = self.donor_insertion

        donor_SNV_summary_string = self.donor_al_SNV_summary(details['candidate_alignment'])
        has_donor_SNV = self.specific_to_donor(details['candidate_alignment'])

        outcome = LongTemplatedInsertionOutcome('donor',
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
                                                donor_SNV_summary_string,
                                               )

        relevant_alignments = details['full_alignments']

        sequencing_direction = self.target_info.sequencing_direction
        sgRNA_strand = self.target_info.sgRNA_feature.strand

        if sequencing_direction == '-' and sgRNA_strand == '+':
            donor_end = outcome.left_insertion_ref_bound
            target_start = outcome.left_target_ref_bound

        elif sequencing_direction == '-' and sgRNA_strand == '-':
            donor_end = outcome.right_insertion_ref_bound
            target_start = outcome.right_target_ref_bound

        elif sequencing_direction == '+' and sgRNA_strand == '+':
            donor_end = outcome.right_insertion_ref_bound
            target_start = outcome.right_target_ref_bound

        elif sequencing_direction == '+' and sgRNA_strand == '-':
            donor_end = outcome.left_insertion_ref_bound
            target_start = outcome.left_target_ref_bound

        else:
            raise NotImplementedError

        # 21.02.22 temp fix
        if target_start is None:
            target_start = len(self.target_info.target_sequence)

        # paired_point represents a donor and target position that
        # should be matched with each other in the intended annealing.
        paired_point = self.target_info.offset_to_HA_ref_ps[0]
        donor_paired_point = paired_point['donor', 'PAM-distal']
        target_paired_point = paired_point['target', 'PAM-distal']
        
        # extension always in negative direction in donor coords
        donor_offset = donor_paired_point - donor_end

        # extension direction in target coords depends on sgRNA strand
        # paired_point is matched and we want offset to be difference from 
        # switching right after matched
        if sgRNA_strand == '+':
            target_offset = target_start - (target_paired_point + 1)
        else:
            # Note: untested if off by 1 here.
            target_offset = (target_paired_point - 1) - target_start

        offset_from_intended_annealing = target_offset - donor_offset

        if self.donor_insertion_matches_intended:
            if details['longest_edge_deletion'] is None or details['longest_edge_deletion'].length <= 2:
                self.category = 'intended edit'
                self.subcategory = 'insertion'
            else:
                self.category = 'edit + indel'
                self.subcategory = 'edit + deletion'

                HDR_outcome = HDROutcome(self.donor_SNV_string, self.donor_deletions_seen)
                deletion_outcome = DeletionOutcome(details['longest_edge_deletion'])
                outcome = HDRPlusDeletionOutcome(HDR_outcome, deletion_outcome)

        # Insist that the extension is a close enough match to the target that
        # the target alignment doesn't get split.
        elif np.abs(offset_from_intended_annealing) <= 1 and self.single_read_covering_target_alignment:
            self.category = 'extension from intended annealing'
            self.subcategory = 'n/a'

        else:
            self.category = 'unintended annealing of RT\'ed sequence'
            if self.alignment_scaffold_overlap(details['cropped_candidate_alignment']) >= 2:
                self.subcategory = 'includes scaffold'
            else:
                self.subcategory = 'no scaffold'

            if not has_donor_SNV:
                self.subcategory += ', no SNV'

            if details['longest_edge_deletion'] is not None:
                self.subcategory += ', with deletion'

        self.relevant_alignments = relevant_alignments
        self.special_alignment = details['candidate_alignment']

        self.outcome = outcome
        self.details = str(outcome)
        
    @memoized_property
    def donor_insertion_matches_intended(self):
        matches = False

        donor_al = self.donor_insertion['candidate_alignment']
        indels = self.extract_indels_from_alignments([donor_al])

        for feature in self.target_info.donor_insertions:
            shares_both_HAs = (self.donor_insertion['shared_HAs'] == {'left', 'right'})
            overlaps_feature = sam.overlaps_feature(donor_al, feature, require_same_strand=False)
            no_big_indels = not any(indel.length > 1 for indel, _ in indels)

            matches = shares_both_HAs and overlaps_feature and no_big_indels
        
        return matches

    def register_SD_MMEJ(self):
        details = self.SD_MMEJ

        self.category = 'SD-MMEJ'
        self.subcategory = details['kind']

        fields = [
            details['left edge'],
            details['gap edges']['left'],
            details['gap edges']['right'],
            details['right edge'],
            details['gap length'],
            details['homology lengths']['left'],
            details['homology lengths']['right'],
        ]
        self.details = ','.join(str(f) for f in fields)

        self.relevant_alignments = self.SD_MMEJ['alignments']

    @memoized_property
    def donor_deletions_seen(self):
        seen = [d for d, _ in self.indels if d.kind == 'D' and d in self.target_info.donor_deletions]
        return seen

    @memoized_property
    def non_donor_deletions(self):
        non_donor_deletions = [d for d, _ in self.indels if d.kind == 'D' and d not in self.target_info.donor_deletions]
        return non_donor_deletions

    def shared_HAs(self, donor_al, target_edge_als):
        q_to_HA_offsets = defaultdict(lambda: defaultdict(set))

        if self.target_info.homology_arms is None:
            return None

        for side in ['left', 'right']:
            # Only register the left-most occurence of the left HA in the left target edge alignment
            # and right-most occurence of the right HA in the right target edge alignment.
            all_q_to_HA_offset = {}
            
            al = target_edge_als[side]

            if al is None or al.is_unmapped:
                continue
            
            for q, read_b, ref_i, ref_b, qual in sam.aligned_tuples(al, self.target_info.target_sequence):
                if q is None:
                    continue
                offset = self.target_info.HA_ref_p_to_offset['target', side].get(ref_i)
                if offset is not None:
                    all_q_to_HA_offset[q] = offset
            
            if side == 'left':
                get_most_extreme = min
                direction = 1
            else:
                get_most_extreme = max
                direction = -1
                
            if len(all_q_to_HA_offset) > 0:
                q = get_most_extreme(all_q_to_HA_offset)
                while q in all_q_to_HA_offset:
                    q_to_HA_offsets[q][side, all_q_to_HA_offset[q]].add('target')
                    q += direction

        if donor_al is None or donor_al.is_unmapped:
            pass
        else:
            for q, read_b, ref_i, ref_b, qual in sam.aligned_tuples(donor_al, self.target_info.donor_sequence):
                if q is None:
                    continue
                for side in ['left', 'right']:
                    offset = self.target_info.HA_ref_p_to_offset['donor', side].get(ref_i)
                    if offset is not None:
                        q_to_HA_offsets[q][side, offset].add('donor')
        
        shared = set()
        for q in q_to_HA_offsets:
            for side, offset in q_to_HA_offsets[q]:
                if len(q_to_HA_offsets[q][side, offset]) == 2:
                    shared.add(side)

        return shared

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

    @memoized_property
    def possible_templated_insertions(self):
        # Make some shorter aliases.
        ti = self.target_info
        edge_als = self.get_target_edge_alignments(self.parsimonious_target_alignments, split=False)

        if edge_als['left'] is not None:
            # If a target alignment to the start of the read exists,
            # insist that it be to the sequencing primer. 
            if not sam.overlaps_feature(edge_als['left'], ti.primers_by_side_of_read['left']):
                return [{'failed': 'left edge alignment isn\'t to primer'}]

        candidates = []
        for donor_al in self.split_donor_alignments:
            candidates.append((donor_al, 'donor'))

        for genomic_al in self.nonredundant_supplemental_alignments:
            candidates.append((genomic_al, 'genomic'))

        possible_insertions = []

        for candidate_al, source in candidates:
            details = {'source': source}

            if source == 'donor':
                candidate_ref_seq = ti.donor_sequence
            else:
                candidate_ref_seq = None
            
            # Find the locations on the query at which switching from edge alignments to the
            # candidate and then back again minimizes the edit distance incurred.
            left_results = sam.find_best_query_switch_after(edge_als['left'], candidate_al, ti.target_sequence, candidate_ref_seq, min)
            right_results = sam.find_best_query_switch_after(candidate_al, edge_als['right'], candidate_ref_seq, ti.target_sequence, max)

            # For genomic insertions, parsimoniously assign maximal query to candidates that make it all the way to the read edge
            # even if there is a short target alignmnent at the edge.
            if source == 'genomic':
                min_left_results = sam.find_best_query_switch_after(edge_als['left'], candidate_al, ti.target_sequence, candidate_ref_seq, min)
                if min_left_results['switch_after'] == -1:
                    left_results = min_left_results

                max_right_results = sam.find_best_query_switch_after(candidate_al, edge_als['right'], candidate_ref_seq, ti.target_sequence, max)
                if max_right_results['switch_after'] == len(self.seq) - 1:
                    right_results = max_right_results

            # TODO: might be valuable to replace max in left_results tie breaker with something that picks the cut site if possible.

            # Crop the alignments at the switch points identified.
            target_bounds = {}
            target_query_bounds = {}

            cropped_left_al = sam.crop_al_to_query_int(edge_als['left'], -np.inf, left_results['switch_after'])
            target_bounds[5] = sam.reference_edges(cropped_left_al)[3]
            target_query_bounds[5] = interval.get_covered(cropped_left_al).end

            cropped_right_al = sam.crop_al_to_query_int(edge_als['right'], right_results['switch_after'] + 1, np.inf)
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
                details['edge_als'] = edge_als
                details['candidate_al'] = candidate_al
                details['switch_afters'] = {'left': left_results['switch_after'], 'right': right_results['switch_after']}
                details['failed'] = 'cropping eliminates insertion'
                possible_insertions.append(details)
                continue

            insertion_reference_bounds = sam.reference_edges(cropped_candidate_al)   
            insertion_query_interval = interval.get_covered(cropped_candidate_al)
            insertion_length = len(insertion_query_interval)
                
            left_edits = sam.edit_distance_in_query_interval(cropped_left_al, ref_seq=ti.target_sequence)
            right_edits = sam.edit_distance_in_query_interval(cropped_right_al, ref_seq=ti.target_sequence)
            middle_edits = sam.edit_distance_in_query_interval(cropped_candidate_al, ref_seq=candidate_ref_seq)
            edit_distance = left_edits + middle_edits + right_edits

            gap_before_length = left_results['gap_length']
            gap_after_length = right_results['gap_length']
            total_gap_length = gap_before_length + gap_after_length
            
            has_donor_SNV = {
                'left': self.specific_to_donor(cropped_left_al),
                'right': self.specific_to_donor(cropped_right_al),
            }
            if source == 'donor':
                has_donor_SNV['insertion'] = self.specific_to_donor(candidate_al) # should this be cropped_candidate_al?

            missing_from_blunt = {
                5: None,
                3: None,
            }

            # TODO: there are edge cases where short extra sequence appears at start of read before expected NotI site.
            if target_bounds[5] is not None:
                if edge_als['left'].is_reverse:
                    missing_from_blunt[5] = target_bounds[5] - (ti.cut_after + 1)
                else:
                    missing_from_blunt[5] = ti.cut_after - target_bounds[5]

            if target_bounds[3] is not None:
                missing_from_blunt[3] = (target_bounds[3] - ti.cut_after)
                if edge_als['right'].is_reverse:
                    missing_from_blunt[3] = ti.cut_after - target_bounds[3]
                else:
                    missing_from_blunt[3] = target_bounds[3] - (ti.cut_after + 1)

            longest_edge_deletion = None

            for side in ['left', 'right']:
                if edge_als[side] is not None:
                    indels = self.extract_indels_from_alignments([edge_als[side]])
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
                'edge_alignments': edge_als,
                'full_alignments': [al for al in [edge_als['left'], candidate_al, edge_als['right']] if al is not None],

                'longest_edge_deletion': longest_edge_deletion,

                'has_donor_SNV': has_donor_SNV,

                'strand': sam.get_strand(candidate_al),
            })

            if source == 'genomic':
                organism, original_name = cropped_candidate_al.reference_name.split('_', 1)
                organism_matches = {n for n in self.target_info.supplemental_headers if cropped_candidate_al.reference_name.startswith(n)}
                if len(organism_matches) != 1:
                    raise ValueError(cropped_candidate_al.reference_name, self.target_info.supplemental_headers)
                else:
                    organism = organism_matches.pop()
                    original_name = cropped_candidate_al.reference_name[len(organism) + 1:]

                header = self.target_info.supplemental_headers[organism]
                al_dict = cropped_candidate_al.to_dict()
                al_dict['ref_name'] = original_name
                original_al = pysam.AlignedSegment.from_dict(al_dict, header)

                details.update({
                    'chr': original_al.reference_name,
                    'organism': organism,
                    'original_alignment': original_al,
                })

                # Since genomic insertions draw from a much large reference sequence
                # than donor insertions, enforce a more stringent minimum length.

                if insertion_length <= 25:
                    details['failed'] = f'insertion length = {insertion_length}'

            if source == 'donor':
                shared_HAs = self.shared_HAs(candidate_al, edge_als)
                details['shared_HAs'] = shared_HAs

                # Only want to handle RT extensions here, which should have PAM-distal homology arm usage,
                # unless the read isn't long enough to include this.
                distance_from_end = self.whole_read.end - details['insertion_query_bounds'][3]
                necessary_read_side = ti.PAM_side_to_read_side['PAM-distal']
                if distance_from_end > 0 and necessary_read_side not in shared_HAs:
                    details['failed'] = f'doesn\'t share PAM-distal HA'

            failures = []

            if gap_before_length > 5:
                failures.append(f'gap_before_length = {gap_before_length}')

            if gap_after_length > 5:
                failures.append(f'gap_after_length = {gap_after_length}')

            if self.error_corrected:
                max_allowable_edit_distance = 5
            else:
                max_allowable_edit_distance = 10

            # Allow a high edit distance if it is almost entirely explained by a single large deletion.
            if edit_distance_besides_deletion > max_allowable_edit_distance:
                failures.append(f'edit_distance = {edit_distance}')

            if has_donor_SNV['left']:
                failures.append('left alignment has a donor SNV')

            if has_donor_SNV['right']:
                failures.append('right alignment has a donor SNV')

            #if insertion_length < 5:
            #    failures.append(f'insertion length = {insertion_length}')

            edit_distance_over_length = middle_edits / insertion_length
            if edit_distance_over_length >= 0.1:
                failures.append(f'edit distance / length = {edit_distance_over_length}')

            if len(failures) > 0:
                details['failed'] = ' '.join(failures)

            possible_insertions.append(details)

        return possible_insertions

    @memoized_property
    def multipart_templated_insertion(self):
        candidates = self.possible_templated_insertions()
        edge_als = self.target_edge_alignments

        lefts = [c for c in candidates if 'gap_before' in c and c['gap_before'] is None]
        rights = [c for c in candidates if 'gap_after' in c and c['gap_after'] is None]
        
        covering_pairs = []
        
        for left in lefts:
            left_al = left['candidate_alignment']
            for right in rights:
                right_al = right['candidate_alignment']
                
                gap_length = right['gap_right_query_edge'] - left['gap_left_query_edge'] - 1

                left_covered = interval.get_covered(left_al)
                right_covered = interval.get_covered(right_al)
                pair_overlaps = left_covered.end >= right_covered.start - 1

                if pair_overlaps and gap_length > 5: 
                    if left['source'] == 'donor':
                        left_ref = self.target_info.donor_sequence
                    else:
                        left_ref = None

                    if right['source'] == 'donor':
                        right_ref = self.target_info.donor_sequence
                    else:
                        right_ref = None

                    switch_results = sam.find_best_query_switch_after(left_al, right_al, left_ref, right_ref, min)
                    
                    cropped_left_al = sam.crop_al_to_query_int(left_al, -np.inf, switch_results['switch_after'])
                    cropped_right_al = sam.crop_al_to_query_int(right_al, switch_results['switch_after'] + 1, np.inf)

                    left_edits = sam.edit_distance_in_query_interval(cropped_left_al, ref_seq=left_ref)
                    right_edits = sam.edit_distance_in_query_interval(cropped_right_al, ref_seq=right_ref)
                    edit_distance = left_edits + right_edits
                    
                    pair_results = {
                        'left_alignment': left['candidate_alignment'],
                        'right_alignment': right['candidate_alignment'],
                        'edit_distance': edit_distance,
                        'left_source': left['source'],
                        'right_source': right['source'],
                        'full_alignments': [al for al in [edge_als['left'], left['candidate_alignment'], right['candidate_alignment'], edge_als['right']] if al is not None],
                        'gap_length': gap_length,
                    }
                    
                    covering_pairs.append(pair_results)
        
        covering_pairs = sorted(covering_pairs, key=lambda p: (p['edit_distance']))
        
        return covering_pairs

    @memoized_property
    def no_alignments_detected(self):
        return all(al.is_unmapped for al in self.alignments)

    @memoized_property
    def perfect_edge_alignments_from_scratch(self):
        seed_length = 10
        left_als = self.seed_and_extend('target', 0, seed_length)
        right_als = self.seed_and_extend('target', len(self.seq) - 1 - seed_length, len(self.seq) - 1)
        return left_als + right_als

    def categorize(self):
        self.outcome = None
        self.trust_inferred_length = True

        if len(self.seq) <= self.target_info.combined_primer_length + 10:
            self.category = 'nonspecific amplification'
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
                if self.has_donor_SNV:
                    self.category = 'intended edit'

                    HDR_outcome = HDROutcome(self.donor_SNV_string, self.donor_deletions_seen)
                    self.outcome = HDR_outcome

                    if len(self.non_donor_SNVs) == 0 and len(uninteresting_indels) == 0:
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
                    self.relevant_alignments = [target_alignment] + self.donor_alignments

                else:
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

            elif self.max_scaffold_overlap >= 2 and self.donor_insertion is not None:
                self.register_donor_insertion()

            elif len(interesting_indels) == 1:
                indel = interesting_indels[0]

                if len(self.donor_deletions_seen) == 1:
                    self.category = 'intended edit'
                    if len(self.non_donor_SNVs) > 0:
                        self.subcategory = 'deletion + unintended mismatches'
                    elif self.has_donor_SNV:
                        self.subcategory = 'deletion + SNV'
                    else:
                        self.subcategory = 'deletion'
                    self.outcome = HDROutcome(self.donor_SNV_string, self.donor_deletions_seen)
                    self.details = str(self.outcome)

                    self.relevant_alignments = [target_alignment] + self.donor_alignments

                else: # one indel, not a donor deletion
                    if self.has_donor_SNV:
                        if indel.kind == 'D':
                            # If the deletion overlaps with HA_RT on the target, consider this an unintended donor integration.                            
                            #if self.deletion_overlaps_HA_RT(indel) and self.donor_insertion is not None:
                            #    self.register_donor_insertion()
                            self.category = 'edit + indel'
                            self.subcategory = 'edit + deletion'
                            HDR_outcome = HDROutcome(self.donor_SNV_string, self.donor_deletions_seen)
                            deletion_outcome = DeletionOutcome(indel)
                            self.outcome = HDRPlusDeletionOutcome(HDR_outcome, deletion_outcome)
                            self.details = str(self.outcome)
                            self.relevant_alignments = self.parsimonious_target_alignments + self.donor_alignments
                        elif indel.kind == 'I' and indel.length == 1:
                            self.category = 'edit + indel'
                            self.subcategory = 'edit + insertion'
                            HDR_outcome = HDROutcome(self.donor_SNV_string, self.donor_deletions_seen)
                            insertion_outcome = InsertionOutcome(indel)
                            self.outcome = HDRPlusInsertionOutcome(HDR_outcome, insertion_outcome)
                            self.details = str(self.outcome)
                            self.relevant_alignments = self.parsimonious_target_alignments + self.donor_alignments
                        else:
                            self.category = 'uncategorized'
                            self.subcategory = 'donor SNV with non-donor indel'
                            self.details = 'n/a'
                            self.relevant_alignments = self.uncategorized_relevant_alignments

                    else: # no donor SNVs
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

                elif len(self.indels) == 2:
                    if len(self.donor_deletions_seen) == 1 and len(self.non_donor_deletions) == 1:
                        self.category = 'edit + indel'
                        self.subcategory = 'edit + deletion'
                        HDR_outcome = HDROutcome(self.donor_SNV_string, self.donor_deletions_seen)
                        indel = self.non_donor_deletions[0]
                        deletion_outcome = DeletionOutcome(indel)
                        self.outcome = HDRPlusDeletionOutcome(HDR_outcome, deletion_outcome)
                        self.details = str(self.outcome)
                        self.relevant_alignments = self.parsimonious_target_alignments + self.donor_alignments

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

        elif self.donor_insertion is not None:
            self.register_donor_insertion()

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

        elif self.genomic_insertion is not None:
            self.register_genomic_insertion()

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

        return self.category, self.subcategory, self.details, self.outcome

    @memoized_property
    def target_multiple_gap_covering_alignments(self):
        initial_target_als = copy.copy(self.split_target_alignments)

        if self.perfect_edge_alignments['right'] is not None:
            initial_target_als.append(self.perfect_edge_alignments['right'])

        initial_uncovered = self.whole_read - interval.get_disjoint_covered(initial_target_als)
        gap_covers = []
        for uncovered_interval in initial_uncovered.intervals:
            # Don't try to explain tiny gaps.
            if len(uncovered_interval) > 4:
                # Note the + 1 on end here.
                gap_covers.extend(self.seed_and_extend('target', uncovered_interval.start, uncovered_interval.end + 1))
        return interval.make_parsimonious(initial_target_als + gap_covers)

    @memoized_property
    def gap_covering_alignments(self):
        ti = self.target_info

        initial_als = copy.copy(self.split_target_and_donor_alignments)

        if self.perfect_edge_alignments['right'] is not None:
            initial_als.append(self.perfect_edge_alignments['right'])

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

            als = [sw.extend_alignment(al, ti.target_sequence_bytes) for al in als]
            
            gap_covers.extend(als)

            if ti.donor is not None:
                als = sw.align_read(self.read,
                                    [(ti.donor, ti.donor_sequence),
                                    ],
                                    4,
                                    ti.header,
                                    N_matches=False,
                                    max_alignments_per_target=5,
                                    read_interval=extended_gap,
                                    mismatch_penalty=-2,
                                )

                als = [sw.extend_alignment(al, ti.donor_sequence_bytes) for al in als]
                
                gap_covers.extend(als)

        all_als = initial_als + gap_covers

        return sam.make_nonredundant(interval.make_parsimonious(all_als))

    @memoized_property
    def target_gap_covering_alignments(self):
        return [al for al in self.gap_covering_alignments if al.reference_name == self.target_info.target]

    @memoized_property
    def donor_gap_covering_alignments(self):
        return [al for al in self.gap_covering_alignments if al.reference_name == self.target_info.donor]

    @memoized_property
    def target_and_donor_multiple_gap_covering_alignments(self):
        initial_als = self.split_target_alignments + self.split_donor_alignments

        if self.perfect_edge_alignments['right'] is not None:
            initial_als.append(self.perfect_edge_alignments['right'])

        initial_uncovered = self.whole_read - interval.get_disjoint_covered(initial_als)
        gap_covers = []
        for uncovered_interval in initial_uncovered.intervals:
            # Don't try to explain tiny gaps.
            if len(uncovered_interval) > 4:
                # Note the + 1 on end here.
                gap_covers.extend(self.seed_and_extend('target', uncovered_interval.start, uncovered_interval.end + 1))

        return interval.make_parsimonious(initial_als + gap_covers)

    @memoized_property
    def long_duplication(self):
        ''' (duplication, simple)   - a single junction
            (duplication, iterated) - multiple uses of the same junction
            (duplication, complex)  - multiple junctions that are not exactly the same
        '''
        ti = self.target_info
        # Order target als by position on the query from left to right.
        target_als = sorted(self.target_gap_covering_alignments, key=interval.get_covered)

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
    
        covered = interval.DisjointIntervals(interval.make_disjoint(covereds))

        uncovered = self.whole_read_minus_edges(2) - covered
        
        if len(merged_als) == 1 or uncovered.total_length > 0:
            return None
        
        ref_junctions = []

        indels = []

        als_with_donor_SNVs = sum(self.specific_to_donor(al) for al in merged_als)

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

        return subcategory, ref_junctions, indels, als_with_donor_SNVs, merged_als

    @memoized_property
    def longest_polyG(self):
        locations = utilities.homopolymer_lengths(self.seq, 'G')

        if locations:
            max_length = max(length for p, length in locations)
        else:
            max_length = 0

        return max_length

    @memoized_property
    def uncategorized_relevant_alignments(self):
        return self.gap_covering_alignments + self.donor_alignments + interval.make_parsimonious(self.nonredundant_supplemental_alignments + self.extra_alignments)

    @memoized_property
    def inferred_amplicon_length(self):
        if self.seq  == '':
            return 0

        right_al = self.get_target_edge_alignments(self.target_gap_covering_alignments)['right']

        if right_al is None or not self.trust_inferred_length:
            inferred_length = len(self.seq)
        else:
            right_al_edge = sam.reference_edges(right_al)[3]

            right_primer = self.target_info.primers_by_side_of_read['right']

            if self.target_info.sequencing_direction == '+':
                right_primer_edge = right_primer.end
                inferred_extra = right_primer_edge - right_al_edge
            else:
                right_primer_edge = right_primer.start
                inferred_extra = right_al_edge - right_primer_edge

            inferred_length = len(self.seq) + max(inferred_extra, 0)

        return inferred_length