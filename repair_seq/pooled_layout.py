from collections import Counter, defaultdict
from itertools import product

import numpy as np
import pysam

from hits import interval, sam, utilities, sw, fastq
from hits.utilities import memoized_property

from knock_knock.target_info import DegenerateDeletion, DegenerateInsertion, SNV, SNVs
from knock_knock import layout

from knock_knock.outcome import *

side_categories = [
    'intended',
    'unintended',
    'blunt',
    'concatamer',
    'ambiguous',
]

donor_misintegration_subcategories = []

for l_cat, r_cat in product(side_categories, repeat=2):
    subcat = f'left {l_cat}, right {r_cat}'
    donor_misintegration_subcategories.append(subcat)
    subcat_no_SNVs = f'{subcat} (no SNVs)'
    donor_misintegration_subcategories.append(subcat_no_SNVs)

donor_misintegration_subcategories = tuple(donor_misintegration_subcategories)

class Layout(layout.Categorizer):
    category_order = [
        ('wild type',
            ('clean',
             'mismatches',
             'indels',
            ),
        ),
        ('mismatches',
            ('mismatches',
            ),
        ),
        ('donor',
            ('clean',
             'synthesis errors',
             'deletion',
             'other',
             'collapsed',
            ),
        ),
        ('donor + deletion',
            ('donor + deletion',
            ),
        ),
        ('donor + insertion',
            ('donor + insertion',
            ),
        ),
        ('deletion',
            ('near cut',
             'far from cut',
            ),
        ),
        ('insertion with deletion',
            ('n/a',
            ),
        ),
        ('deletion + mismatches',
            ('deletion + mismatches',
            ),
        ),
        ('multiple deletions',
            ('multiple deletions',
            ),
        ),
        ('insertion',
            ('near cut',
            'far from cut',
            ),
        ),
        ('insertion + mismatches',
            ('insertion + mismatches',
            ),
        ),
        ('duplication',
            ('simple',
             'iterated',
             'complex',
            ),
        ),
        ('SD-MMEJ',
            ('loop-out',
             'snap-back',
             'multi-step',
            ),
        ),
        ('genomic insertion',
            ('hg19',
             'bosTau7',
             'e_coli',
            ),
        ),
        ('transversion',
            ('NotI',
             'non-NotI',
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
        ('donor misintegration',
            donor_misintegration_subcategories,
        ),
        ('blunt misintegration',
            ("5' HDR, 3' blunt",
             "5' blunt, 3' HDR",
             "5' blunt, 3' blunt",
             "5' blunt, 3' imperfect",
             "5' imperfect, 3' blunt",
            ),
        ),
        ('incomplete HDR',
            ("5' HDR, 3' imperfect",
             "5' imperfect, 3' HDR",
             "5' imperfect, 3' imperfect",
             "5' HDR, 3' HDR",
            ),
        ),
        ('complex misintegration',
            ("5' imperfect, 3' imperfect",
            'other',
            ),
        ),
        ('complex templated insertion',
            ('genomic',
             'donor',
             'mixed',
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
             'donor SNV with non-donor insertion',
             'one indel plus more than one mismatch',
             'multiple indels',
             'multiple indels plus at least one mismatch',
             'multiple indels including donor deletion',
             'multiple indels plus at least one donor SNV',
            ),
        ),
        ('bad sequence',
            ('low quality',
             'no alignments detected',
             'non-overlapping',
            ),
        ), 
    ]

    def __init__(self, alignments, target_info, error_corrected=True, mode='cutting'):
        self.alignments = [al for al in alignments if not al.is_unmapped]
        self.target_info = target_info
        
        alignment = alignments[0]
        self.query_name = alignment.query_name
        self.seq = sam.get_original_seq(alignment)

        if self.seq is not None:
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

        self.inferred_amplicon_length = len(self.seq) if self.seq is not None else -1

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
    def primary_alignments(self):
        p_als = [
            al for al in self.alignments
            if al.reference_name in self.primary_ref_names
        ]
        
        return p_als
    
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

    @memoized_property
    def merged_perfect_edge_alignments(self):
        ''' If the read can be explain as perfect alignments from each end that cover the whole query,
        return a single (possibly deletion containing) alignment that does this.
        '''
        edge_als = self.perfect_edge_alignments

        if self.covers_whole_read(edge_als['left']):
            merged_al = edge_als['left']
        elif self.covers_whole_read(edge_als['right']):
            merged_al = edge_als['right']
        else:
            merged_al = sam.merge_adjacent_alignments(edge_als['left'], edge_als['right'], self.target_info.reference_sequences)

            if merged_al is None and edge_als['right'] is not None:
                extended_right_al = sw.extend_alignment_with_one_nt_deletion(edge_als['right'], self.target_info.target_sequence_bytes)
                if self.covers_whole_read(extended_right_al):
                    merged_al = extended_right_al
                else:
                    merged_al = sam.merge_adjacent_alignments(edge_als['left'], extended_right_al, self.target_info.reference_sequences)

        return merged_al
    
    @memoized_property
    def parsimonious_target_alignments(self):
        if self.merged_perfect_edge_alignments is not None:
            # If it is possible to explain the read as a single deletion, do it.
            als = [self.merged_perfect_edge_alignments]
        else:
            als = interval.make_parsimonious(self.split_target_alignments)

        if len(als) == 0:
            return als

        if len(als) == 2 and sam.get_strand(als[0]) == sam.get_strand(als[1]):
            upstream, downstream = sorted(als, key=lambda al: al.reference_start)

            merged = sam.merge_adjacent_alignments(upstream, downstream, self.target_info.reference_sequences)
            if merged is not None:
                als = [merged]

        # If the right edge of the read isn't covered, try to merge a perfect edge alignment to the right-most alignment.
        covered = interval.get_disjoint_covered(als)
        if len(self.seq) - 1 not in covered:
            right_most = max(als, key=lambda al: interval.get_covered(al).end)
            other = [al for al in als if al != right_most]

            perfect_edge_als = self.perfect_edge_alignments
            merged = sam.merge_adjacent_alignments(right_most, perfect_edge_als['right'], self.target_info.reference_sequences)
            if merged is None:
                merged = right_most

            als = other + [merged]
        
        if 0 not in covered:
            left_most = min(als, key=lambda al: interval.get_covered(al).start)
            other = [al for al in als if al != left_most]

            perfect_edge_als = self.perfect_edge_alignments
            merged = sam.merge_adjacent_alignments(perfect_edge_als['left'], left_most, self.target_info.reference_sequences)
            if merged is None:
                merged = left_most

            als = other + [merged]
                
        #exempt_if_overlaps = self.target_info.around_cuts(5)
        #split_als = []
        #for al in als:
        #    split_als.extend(sam.split_at_deletions(al, 3, exempt_if_overlaps))
        split_als = als
        
        return split_als
    
    @memoized_property
    def split_target_and_donor_alignments(self):
        all_split_als = []
        for al in self.target_alignments + self.donor_alignments:
            split_als = layout.comprehensively_split_alignment(al, self.target_info, 'illumina', self.ins_size_to_split_at, self.del_size_to_split_at)

            if al.reference_name == self.target_info.target:
                seq_bytes = self.target_info.target_sequence_bytes
            else:
                seq_bytes = self.target_info.donor_sequence_bytes

            extended = [sw.extend_alignment(split_al, seq_bytes) for split_al in split_als]

            all_split_als.extend(extended)

        return all_split_als

    @memoized_property
    def gap_sw_realignments(self):
        ti = self.target_info
        gaps = self.whole_read - interval.get_disjoint_covered(self.split_target_and_donor_alignments)

        all_gap_als = []

        targets = [
            (ti.target, ti.target_sequence),
        ]

        if ti.donor is not None:
            targets.append((ti.donor, ti.donor_sequence))

        try:
            target_window_start = ti.features[ti.target, 'sgRNA constant region E+F'].start
            target_window_end = ti.features[ti.target, 'EF1a'].end + 1

            ref_intervals = {
                ti.target: interval.Interval(target_window_start, target_window_end),
            }
        except KeyError:
            ref_intervals = {
                ti.target: ti.amplicon_interval,
            }

        if ti.donor is not None:
            ref_intervals[ti.donor] = interval.Interval(0, len(ti.donor_sequence) - 1)

        for gap in gaps.intervals:
            start = max(0, gap.start - 5)
            end = min(len(self.seq) - 1, gap.end + 5)
            extended_gap = interval.Interval(start, end)

            als = sw.align_read(self.read,
                                targets,
                                5,
                                ti.header,
                                read_interval=extended_gap,
                                ref_intervals=ref_intervals,
                                max_alignments_per_target=4,
                                mismatch_penalty=-8,
                                indel_penalty=-60,
                                min_score_ratio=0,
                                both_directions=True,
                                N_matches=False,
                               )
            
            extended_target_als = [sw.extend_alignment(al, ti.target_sequence_bytes) for al in als if al.reference_name == ti.target]

            if ti.donor is not None:
                extended_donor_als = [sw.extend_alignment(al, ti.donor_sequence_bytes) for al in als if al.reference_name == ti.donor]
            else:
                extended_donor_als = []
            
            all_gap_als.extend(extended_target_als + extended_donor_als)

        return all_gap_als

    @memoized_property
    def split_donor_alignments(self):
        return [al for al in self.split_target_and_donor_alignments if al.reference_name == self.target_info.donor]
    
    @memoized_property
    def split_target_alignments(self):
        return [al for al in self.split_target_and_donor_alignments if al.reference_name == self.target_info.target]

    @memoized_property
    def target_edge_alignments(self):
        edge_als = self.get_target_edge_alignments(self.parsimonious_target_alignments)

        # If blastn didn't find any alignments to an edge, look for a short perfect alignment.
        for side in ['left', 'right']:
            if edge_als[side] is None:
                edge_als[side] = self.perfect_edge_alignments[side]

        return edge_als

    def get_target_edge_alignments(self, alignments):
        ''' Get target alignments that make it to the read edges. '''
        edge_alignments = {'left': [], 'right':[]}

        all_split_als = []
        for al in alignments:
            split_als = layout.comprehensively_split_alignment(al, self.target_info, 'illumina', self.ins_size_to_split_at, self.del_size_to_split_at)
            
            target_seq_bytes = self.target_info.reference_sequences[al.reference_name].encode()
            extended = [sw.extend_alignment(split_al, target_seq_bytes) for split_al in split_als]

            all_split_als.extend(extended)

        for al in all_split_als:
            if sam.get_strand(al) != self.target_info.sequencing_direction:
                continue

            covered = interval.get_covered(al)
            
            if covered.start <= 2:
                edge_alignments['left'].append(al)
            
            if covered.end >= len(self.seq) - 1 - 2:
                edge_alignments['right'].append(al)

        for edge in ['left', 'right']:
            if len(edge_alignments[edge]) == 0:
                edge_alignments[edge] = None
            else:
                edge_alignments[edge] = max(edge_alignments[edge], key=lambda al: al.query_alignment_length)

        return edge_alignments

    def extends_past_PAS(self, al):
        if (self.target_info.target, 'PAS') in self.target_info.features:
            PAS = self.target_info.features[self.target_info.target, 'PAS']
            return al.reference_start <= PAS.start <= al.reference_end
        else:
            return False

    def overlaps_primer(self, al, side, require_correct_strand=True):
        primer = self.target_info.primers_by_side_of_read[side]
        num_overlapping_bases = al.get_overlap(primer.start, primer.end + 1)
        overlaps = num_overlapping_bases > 0
        correct_strand = sam.get_strand(al) == self.target_info.sequencing_direction 

        return overlaps and (correct_strand or not require_correct_strand)

    @memoized_property
    def whole_read(self):
        return interval.Interval(0, len(self.seq) - 1)

    def whole_read_minus_edges(self, edge_length):
        if self.seq is None:
            return interval.Interval.empty()
        else:
            return interval.Interval(edge_length, len(self.seq) - 1 - edge_length)
    
    @memoized_property
    def single_target_alignment(self):
        t_als = self.parsimonious_target_alignments
        
        if len(t_als) != 1:
            return None
        else:
            return t_als[0]

    @memoized_property
    def query_missing_from_single_target_alignment(self):
        t_al = self.single_target_alignment
        if t_al is None:
            return None
        else:
            split_als = sam.split_at_large_insertions(t_al, 5)
            covered = interval.get_disjoint_covered(split_als)
            ignoring_edges = interval.Interval(covered.start, covered.end)

            missing_from = {
                'start': covered.start,
                'end': len(self.seq) - covered.end - 1,
                'middle': (ignoring_edges - covered).total_length,
            }

            return missing_from

    @memoized_property
    def single_alignment_covers_read(self):
        missing_from = self.query_missing_from_single_target_alignment

        if missing_from is None:
            return False
        else:
            not_too_much = {
                'start': (missing_from['start'] <= 2) or self.overlaps_primer(self.single_target_alignment, 'left'),
                'end': (missing_from['end'] <= 2) or self.overlaps_primer(self.single_target_alignment, 'right'),
                'middle': missing_from['middle'] <= 5,
            }

            return all(not_too_much.values())

    def overlaps_primer(self, al, side):
        primer = self.target_info.primers_by_side_of_read[side]
        num_overlapping_bases = al.get_overlap(primer.start, primer.end + 1)
        overlaps = num_overlapping_bases > 0
        correct_strand = sam.get_strand(al) == self.target_info.sequencing_direction 

        return correct_strand and overlaps

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
        return sam.overlaps_feature(edge_al, self.target_info.sequencing_start)

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
        donor_SNV_locii = {name: [] for name in SNPs['donor']}

        for true_read_i, read_b, ref_i, ref_b, qual in sam.aligned_tuples(donor_al, ref_seq):
            # Note: read_b and ref_b are as if the read is the forward strand
            if ref_i in position_to_name:
                name = position_to_name[ref_i]

                # not confident that the logic is right here
                if SNPs['target'][name]['strand'] != SNPs['donor'][name]['strand']:
                    read_b = utilities.reverse_complement(read_b)

                donor_SNV_locii[name].append((read_b, qual))

        return donor_SNV_locii

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
        donor_SNV_locii, _ = self.SNVs_summary
        return self.process_locii_summaries([donor_SNV_locii])

    def process_locii_summaries(self, locii_summaries):
        SNPs = self.target_info.donor_SNVs

        all_summaries = defaultdict(list)
        for locii_summary in locii_summaries:
            for k, v in locii_summary.items():
                all_summaries[k].extend(v)

        genotype = {}

        has_donor_SNV = False
        has_nondonor_SNV_at_donor_position = False

        for name in sorted(SNPs['target']):
            bs = defaultdict(list)

            for b, q in all_summaries[name]:
                bs[b].append(q)

            # If only '-' was seen, its fine to remove it. If a non-deletion was seen, favor it.
            bs.pop('-', None)
            
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
                    else:
                        has_nondonor_SNV_at_donor_position = True

        string_summary = ''.join(genotype[name] for name in sorted(SNPs['target']))

        return has_donor_SNV, has_nondonor_SNV_at_donor_position, string_summary

    @memoized_property
    def has_donor_SNV(self):
        has_donor_SNV, _, _ = self.donor_SNV_locii_summary
        return has_donor_SNV

    @memoized_property
    def has_nondonor_SNV_at_donor_position(self):
        _, has_nondonor_SNV_at_donor_position, _ = self.donor_SNV_locii_summary
        return has_nondonor_SNV_at_donor_position

    @memoized_property
    def has_any_SNV(self):
        return self.has_donor_SNV or (len(self.non_donor_SNVs) > 0)

    @memoized_property
    def donor_SNV_string(self):
        _, _, string_summary = self.donor_SNV_locii_summary
        return string_summary

    def extract_indels_from_alignments(self, als):
        ti = self.target_info

        around_cut_interval = ti.around_cuts(5)

        primer_intervals = interval.make_disjoint([interval.Interval.from_feature(p) for p in ti.primers.values()])

        polyT = ti.features.get((ti.target, 'polyT-track'))
        if polyT is not None:
            polyT_interval = interval.Interval(polyT.start, polyT.end)
        else:
            polyT_interval = interval.Interval.empty()

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

                # TODO: indel_interval should be calculated after expand_degenerate 
                near_cut = len(indel_interval & around_cut_interval) > 0
                entirely_in_primer = indel_interval in primer_intervals
                entirely_in_polyT = indel_interval in polyT_interval

                indel = self.target_info.expand_degenerate_indel(indel)
                indels.append((indel, near_cut, entirely_in_primer, entirely_in_polyT))

        # Ignore any indels entirely contained in primers.

        indels = [(indel, near_cut) for indel, near_cut, entirely_in_primer, entirely_in_polyT in indels if not entirely_in_primer and not entirely_in_polyT]

        return indels

    @memoized_property
    def indels(self):
        return self.extract_indels_from_alignments(self.parsimonious_target_alignments)

    @memoized_property
    def indels_near_cut(self):
        return [indel for indel, near_cut in self.indels if near_cut]

    @memoized_property
    def indels_string(self):
        if len(self.indels) > 1:
            raise NotImplementedError
        return str(self.indels[0][0])

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
    def not_covered_by_target_or_donor_plus_realigned(self):
        covered = interval.get_disjoint_covered(self.gap_sw_realignments)
        return self.not_covered_by_target_or_donor & (self.whole_read - covered)

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
            novel_covered = covered & self.not_covered_by_target_or_donor_plus_realigned

            if novel_covered.total_length > 10:
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
        primers = ti.primers_by_side_of_read
        if ti.sequencing_direction == '+':
            valid_intervals = {
                'left': interval.Interval(primers['left'].start, ti.cut_after),
                'right': interval.Interval(ti.cut_after + 1, primers['right'].end),
            }
        else:
            valid_intervals = {
                'left': interval.Interval(ti.cut_after + 1, primers['left'].end),
                'right': interval.Interval(primers['right'].start, ti.cut_after),
            }
        return valid_intervals

    @memoized_property
    def valid_strand_for_edge_alignments(self):
        return self.target_info.sequencing_direction

    @memoized_property
    def perfect_edge_alignments_and_gap(self):
        # Set up keys to prioritize alignments.
        # Prioritize by (correct strand and side of cut, length (longer better), distance from cut (closer better)) 
        
        def sort_key(al, side):
            length = al.query_alignment_length

            valid_int = self.valid_intervals_for_edge_alignments[side]
            valid_strand = self.valid_strand_for_edge_alignments

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

        # Insist that the alignments be to the correct side and strand, even if longer ones
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
    def nonoverlapping_perfect_edge_alignments(self):
        ''' Crops perfect edge alignments to not overlap on read. 
        Returns None if edge als are identical (i.e. wild type).
        '''
        edge_als = self.perfect_edge_alignments

        if sam.fingerprint(edge_als['left']) == sam.fingerprint(edge_als['right']):
            return None

        covered_on_read = {side: interval.get_covered(edge_als[side]) for side in ['left', 'right']}

        covered_by_both = covered_on_read['left'] & covered_on_read['right']

        if len(covered_by_both) == 0:
            crop_interval = {
                'left': (0, np.inf),
                'right': (0, np.inf),
            }
        else:
            crop_interval = {
                'left': (0, covered_by_both.start - 1),
                'right': (covered_by_both.start, np.inf),
            }
            
        cropped = {
            side: sam.crop_al_to_query_int(edge_als[side], *crop_interval[side])
            for side in ['left', 'right']
        }

        return cropped

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

    def gap_covering_alignments(self, required_MH_start, required_MH_end, only_close=False):
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
        partial_als['start'], partial_als['end'] = self.partial_gap_perfect_alignments(1, 1)
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
        #for edge in ['left', 'right']:
        #    if overlaps_cut(best_edge_als[edge]):
        #        details['failed'] = f'{edge} edge alignment extends over cut'
        #        return details

        if gap_interval is None:
            details['failed'] = 'no gap' 
            return details

        details['gap length'] = len(gap_interval)

        # Insist on at least 1 nt of MH on each side.
        gap_covering_als = self.gap_covering_alignments(1, 1)

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
        if self.mode == 'base_editing':
            # Mismatches in base editing data lead to overly permissive calling of SD-MMEJ.
            return False
        else:
            return 'failed' not in self.SD_MMEJ

    @memoized_property
    def is_valid_insertion_with_deletion(self):
        if self.mode == 'base_editing':
            # Mismatches in base editing data lead to overly permissive calling of SD-MMEJ.
            return False

        if self.genomic_insertion or self.donor_insertion:
            return False

        status = self.insertion_with_deletion_status

        if status['kind'] in ['insertion', 'deletion', 'insertion with deletion']:
            return True
        else:
            return False

    @memoized_property
    def insertion_with_deletion_status(self):
        edge_als = self.nonoverlapping_perfect_edge_alignments

        if edge_als is None:
            results = {
                'kind': 'wild type',
            }
            return results
        elif edge_als['left'] is None or edge_als['right'] is None:
            results = {
                'kind': 'unknown',
            }
            return results

        covered_on_ref = {side: interval.get_covered_on_ref(edge_als[side]) for side in ['left', 'right']}

        covered_by_both = covered_on_ref['left'] & covered_on_ref['right']

        strand = self.target_info.sequencing_direction

        if len(covered_by_both) == 0:
            crop_interval = {
                'leftmost': {
                    'left': (0, np.inf),
                    'right': (0, np.inf),
                },
                'rightmost': {
                    'left': (0, np.inf),
                    'right': (0, np.inf),
                },
            }
        else:
            # If the edge alignments overlap on the reference, crop back 
            # so that each ref base is only assigned to one.
            if strand == '+':
                crop_interval = {
                    'leftmost': {
                        'left': (0, covered_by_both.end),
                        'right': (covered_by_both.end + 1, np.inf),
                    },
                    'rightmost': {
                        'left': (0, covered_by_both.start - 1),
                        'right': (covered_by_both.start, np.inf),
                    },
                }
            else:
                crop_interval = {
                    'leftmost': {
                        'left': (covered_by_both.end + 1, np.inf),
                        'right': (0, covered_by_both.end),
                    },
                    'rightmost': {
                        'left': (covered_by_both.start, np.inf),
                        'right': (0, covered_by_both.start - 1),
                    },
                }
        
        cropped = {
            extremity: {
                side: sam.crop_al_to_ref_int(edge_als[side], *crop_interval[extremity][side])
                for side in ['left', 'right']
            }
            for extremity in ['leftmost', 'rightmost']
        }

        if cropped['leftmost']['left'] is None or cropped['leftmost']['right'] is None:
            results = {
                'kind': 'unknown',
            }
            return results
        
        if strand == '+':
            deletion_starts_at = cropped['leftmost']['left'].reference_end
            deletion_length = cropped['leftmost']['right'].reference_start - deletion_starts_at
        else:
            deletion_starts_at = cropped['leftmost']['right'].reference_end
            deletion_length = cropped['leftmost']['left'].reference_start - deletion_starts_at

        covered_on_read = {side: interval.get_covered(cropped['leftmost'][side]) for side in ['left', 'right']}
        uncovered = self.whole_read - covered_on_read['left'] - covered_on_read['right']
        insertion = self.read.seq[uncovered.start:uncovered.end + 1]

        if strand == '+':
            insertion_seq = insertion
        else:
            insertion_seq = utilities.reverse_complement(insertion)

        insertion_starts_after = deletion_starts_at - 1

        if deletion_length > 0:
            deletion = DegenerateDeletion([deletion_starts_at], deletion_length)
        else:
            deletion = None

        if len(insertion_seq) > 0:
            insertion = DegenerateInsertion([insertion_starts_after], [insertion_seq])
        else:
            insertion = None

        if deletion is not None and insertion is None:
            deletion = self.target_info.expand_degenerate_indel(deletion)
            deletion_interval = interval.Interval(min(deletion.starts_ats), max(deletion.ends_ats))

            around_cut_interval = self.target_info.around_cuts(5)
            near_cut = len(deletion_interval & around_cut_interval) > 0

            results = {
                'kind': 'deletion',
                'indel': deletion,
                'near_cut': near_cut,
                'relevant_alignments': [cropped['leftmost']['left'], cropped['leftmost']['right']],
            }

        elif deletion is None and insertion is not None:
            insertion = self.target_info.expand_degenerate_indel(insertion)
            insertion_interval = interval.Interval(min(insertion.starts_afters), max(insertion.starts_afters))

            around_cut_interval = self.target_info.around_cuts(5)
            near_cut = len(insertion_interval & around_cut_interval) > 0

            results = {
                'kind': 'insertion',
                'indel': insertion,
                'near_cut': near_cut,
                'relevant_alignments': [cropped['leftmost']['left'], cropped['leftmost']['right']],
            }

        elif deletion is not None and insertion is not None:
            results = {
                'kind': 'insertion with deletion',
                'indel': (insertion, deletion),
                'relevant_alignments': [cropped['leftmost']['left'], cropped['leftmost']['right']],
            }

        else:
            results = {
                'kind': 'unknown',
            }

        return results

    def register_insertion_with_deletion(self):
        results = self.insertion_with_deletion_status

        if results['kind'] in ['insertion', 'deletion']:
            self.register_indel(results['indel'], results['near_cut'], results['relevant_alignments'])
        elif results['kind'] == 'insertion with deletion':
            self.category = 'insertion with deletion'
            self.subcategory = 'n/a'
            insertion, deletion = results['indel']
            self.outcome = InsertionWithDeletionOutcome(InsertionOutcome(insertion),
                                                        DeletionOutcome(deletion),
                                                       ) 
            self.details = str(self.outcome)
        else:
            raise ValueError(results['kind'])

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

        elif outcome.left_insertion_query_bound == 0:
            # Transversion if read starts with genomic insertion
            self.category = 'transversion'

            # Check if sequence at start of read matches NotI restriction site.
            fetcher = self.target_info.genomic_region_fetchers[outcome.source]

            NotI_site = 'GCGGCCGC'

            if outcome.strand == '-':
                start = outcome.left_insertion_ref_bound - 5
                end = outcome.left_insertion_ref_bound + 3
                possible_NotI = fetcher(outcome.ref_name, start, end)
                possible_NotI = utilities.reverse_complement(possible_NotI)
            else:
                start = outcome.left_insertion_ref_bound - 2
                end = outcome.left_insertion_ref_bound + 6
                possible_NotI = fetcher(outcome.ref_name, start, end)

            if possible_NotI == NotI_site:
                self.subcategory = 'NotI'
            else:
                self.subcategory = 'non-NotI'

        else:
            self.category = 'genomic insertion'
            self.subcategory = details['organism']

        self.details = str(outcome)
        self.relevant_alignments = details['full_alignments']
        self.special_alignment = details['cropped_candidate_alignment']

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
        self.relevant_alignments = details['full_alignments'] + details['concatamer_alignments']

        if 'left' in details['shared_HAs']:
            left_category = 'intended'
        elif details['insertion_reaches_reference_edge']['left']:
            left_category = 'blunt'
        else:
            left_category = 'unintended'

        if details['insertion_reaches_read_edge']:
            right_category = 'ambiguous'
        elif 'right' in details['shared_HAs']:
            right_category = 'intended'
        elif len(details['concatamer_alignments']) > 0:
            right_category = 'concatamer'
        elif details['insertion_reaches_reference_edge']['right']:
            right_category = 'blunt'
        else:
            right_category = 'unintended'

        from_target_SNV_locii, _ = self.SNVs_summary
        from_donor_SNV_locii = self.donor_al_SNV_summary(details['candidate_alignment'])

        # 2020.10.16:  don't remember why target was added here. Removing it.
        _, _, donor_SNV_summary_string = self.process_locii_summaries([from_donor_SNV_locii])

        # Short deletions, especially next to a donor SNV, can cause outcomes that we want
        # to be considered (donor, synthesis error) outcomes to end up here. Triage them here. 
        if left_category == 'intended' and right_category == 'intended':
            al = details['candidate_alignment']
            indels = self.extract_indels_from_alignments([al])
            indels_besides_one_base_deletions = [indel for indel, _ in indels if not (indel.kind == 'D' and indel.length == 1)]

            if len(indels_besides_one_base_deletions) == 0:
                self.category = 'donor'

                if len(indels) > 0:
                    self.subcategory = 'synthesis errors'
                else:
                    self.subcategory = 'clean'

                self.outcome = HDROutcome(donor_SNV_summary_string, self.donor_deletions_seen)
                self.details = str(self.outcome)

                return

        # If we make it past the return above, this read was not triaged. 

        subcategory = f'left {left_category}, right {right_category}'

        if not details['has_donor_SNV']['insertion']:
            subcategory += ' (no SNVs)'

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

        self.outcome = outcome
        self.details = str(outcome)
        
        self.category = 'donor misintegration'
        self.subcategory = subcategory
        self.special_alignment = details['candidate_alignment']

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

    def register_indel(self, indel, near_cut, relevant_alignments=None):
        if indel.kind == 'D':
            self.category = 'deletion'
            self.outcome = DeletionOutcome(indel)
        elif indel.kind == 'I':
            self.category = 'insertion'
            self.outcome = InsertionOutcome(indel)
        else:
            raise ValueError

        if near_cut:
            self.subcategory = 'near cut'
        else:
            self.subcategory = 'far from cut'

        self.details = str(indel)
        if relevant_alignments is None:
            relevant_alignments = [self.single_target_alignment]
        self.relevant_alignments = relevant_alignments

    def register_deletion_plus_mismatches(self):
        self.category = 'deletion + mismatches'
        self.subcategory = 'deletion + mismatches'

        indel, near_cut = self.indels[0]
        if indel.kind != 'D':
            raise ValueError

        deletion_outcome = DeletionOutcome(indel)
        mismatch_outcome = MismatchOutcome(self.non_donor_SNVs)
        self.outcome = DeletionPlusMismatchOutcome(deletion_outcome, mismatch_outcome)
        self.details = str(self.outcome)

        self.relevant_alignments = self.parsimonious_target_alignments

    def register_insertion_plus_mismatches(self):
        self.category = 'insertion + mismatches'
        self.subcategory = 'insertion + mismatches'

        indel, near_cut = self.indels[0]
        if indel.kind != 'I':
            raise ValueError

        insertion_outcome = InsertionOutcome(indel)
        mismatch_outcome = MismatchOutcome(self.non_donor_SNVs)
        self.outcome = InsertionPlusMismatchOutcome(insertion_outcome, mismatch_outcome)
        self.details = str(self.outcome)

        self.relevant_alignments = self.parsimonious_target_alignments

    @memoized_property
    def donor_deletions_seen(self):
        seen = [d for d, _ in self.indels if d.kind == 'D' and d in self.target_info.donor_deletions]
        return seen

    def shared_HAs(self, donor_al, target_edge_als):
        q_to_HA_offsets = defaultdict(lambda: defaultdict(set))

        if self.target_info.homology_arms is None:
            return set()

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
        
        #for q in sorted(q_to_HA_offsets):
        #    print(q, q_to_HA_offsets[q])
            
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
        edge_als = self.target_edge_alignments

        if edge_als['left'] is not None:
            # If a target alignment to the start of the read exists,
            # insist that it be to the sequencing primer. 
            if not sam.overlaps_feature(edge_als['left'], ti.sequencing_start):
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
            left_results = sam.find_best_query_switch_after(edge_als['left'], candidate_al, ti.target_sequence, candidate_ref_seq, max)
            right_results = sam.find_best_query_switch_after(candidate_al, edge_als['right'], candidate_ref_seq, ti.target_sequence, min)

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
                    edges = sam.reference_edges(cropped_right_al)
                    target_bounds[3] = edges[5]
                    target_query_bounds[3] = interval.get_covered(cropped_right_al).start
                else:
                    target_bounds[3] = None
                    target_query_bounds[3] = len(self.seq)

            cropped_candidate_al = sam.crop_al_to_query_int(candidate_al, left_results['switch_after'] + 1, right_results['switch_after'])
            if cropped_candidate_al is None or cropped_candidate_al.is_unmapped:
                details['failed'] = 'cropping eliminates insertion'
                possible_insertions.append(details)
                continue

            strand = sam.get_strand(candidate_al)

            insertion_reference_bounds = sam.reference_edges(cropped_candidate_al)   
            insertion_uncropped_reference_bounds = sam.reference_edges(candidate_al)   

            insertion_query_interval = interval.get_covered(cropped_candidate_al)
            insertion_uncropped_query_interval = interval.get_covered(candidate_al)
            insertion_length = len(insertion_query_interval)

            insertion_reaches_read_edge = self.whole_read.end - insertion_uncropped_query_interval.end <= 2
                
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

            details.update({
                'source': source,
                'insertion_length': insertion_length,

                'insertion_reference_bounds': insertion_reference_bounds,
                'insertion_uncropped_reference_bounds': insertion_uncropped_reference_bounds,

                'insertion_query_bounds': {5: insertion_query_interval.start, 3: insertion_query_interval.end},
                'insertion_reaches_read_edge': insertion_reaches_read_edge,

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
                'candidate_alignment': candidate_al,
                'cropped_candidate_alignment': cropped_candidate_al,
                'target_bounds': target_bounds,
                'target_query_bounds': target_query_bounds,
                'cropped_alignments': [al for al in [cropped_left_al, cropped_candidate_al, cropped_right_al] if al is not None],
                'edge_alignments': edge_als,
                'full_alignments': [al for al in [edge_als['left'], candidate_al, edge_als['right']] if al is not None],

                'has_donor_SNV': has_donor_SNV,

                'strand': strand,
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
                    continue

            if source == 'donor':
                details['shared_HAs'] = self.shared_HAs(candidate_al, edge_als)

                # Post-insertion target alignment on the wrong side of the cut might indicated donor concatamer.
                if target_bounds[3] is None:
                    non_physical_restart = False
                else:
                    if ti.sequencing_start.strand == '-':
                        non_physical_restart = target_bounds[3] > ti.cut_after + 10
                    else:
                        non_physical_restart = target_bounds[3] < ti.cut_after - 10

                concatamer_als = []
                if non_physical_restart:
                    for al in self.split_donor_alignments:
                        possible_concatamer = sam.crop_al_to_query_int(al, insertion_uncropped_query_interval.end + 1, np.inf)
                        if possible_concatamer is not None:
                            concatamer_als.append(possible_concatamer)

                details['concatamer_alignments'] = concatamer_als

                insertion_reaches_reference_edge = {}
                if strand == '+':
                    insertion_reaches_reference_edge['left'] = insertion_uncropped_reference_bounds[5] <= 2
                    insertion_reaches_reference_edge['right'] = len(ti.donor_sequence) - insertion_uncropped_reference_bounds[3] <= 2
                else:
                    insertion_reaches_reference_edge['left'] = len(ti.donor_sequence) - insertion_uncropped_reference_bounds[5] <= 2
                    insertion_reaches_reference_edge['right'] = insertion_uncropped_reference_bounds[3] <= 2

                details['insertion_reaches_reference_edge'] = insertion_reaches_reference_edge

            failures = []

            if gap_before_length > 5:
                failures.append(f'gap_before_length = {gap_before_length}')

            if gap_after_length > 5:
                failures.append(f'gap_after_length = {gap_after_length}')

            if edit_distance > 5:
                failures.append(f'edit_distance = {edit_distance}')

            if has_donor_SNV['left']:
                failures.append('left alignment has a donor SNV')

            if has_donor_SNV['right']:
                failures.append('right alignment has a donor SNV')

            if insertion_length < 5:
                failures.append(f'insertion length = {insertion_length}')

            edit_distance_over_length = middle_edits / insertion_length
            if edit_distance_over_length >= 0.1:
                failures.append(f'edit distance / length = {edit_distance_over_length}')

            if len(failures) > 0:
                details['failed'] = ' '.join(failures)

            possible_insertions.append(details)

        return possible_insertions

    @memoized_property
    def multipart_templated_insertion(self):
        candidates = self.possible_templated_insertions
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

                    # Need to include edits from target alignments so that there is incentive to
                    # choose a donor-SNV-containing donor alignment.

                    left_target_edits = left['left_edits'] 
                    right_target_edits = right['right_edits'] 
                    edit_distance = left_target_edits + left_edits + right_edits + right_target_edits
                    
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
    def one_base_deletions(self):
        return [indel for indel, near_cut in self.indels if indel.kind == 'D' and indel.length == 1]

    @memoized_property
    def indels_besides_one_base_deletions(self):
        return [indel for indel, near_cut in self.indels if not (indel.kind == 'D' and indel.length == 1)]

    @memoized_property
    def perfect_edge_alignments_from_scratch(self):
        seed_length = 10
        left_als = self.seed_and_extend('target', 0, seed_length)
        right_als = self.seed_and_extend('target', len(self.seq) - 1 - seed_length, len(self.seq) - 1)
        return left_als + right_als

    @memoized_property
    def duplication(self):
        ''' (duplication, simple)   - a single junction
            (duplication, iterated) - multiple uses of the same junction
            (duplication, complex)  - multiple junctions that are not exactly the same
        '''
        ti = self.target_info
        # Order target als by position on the query from left to right.
        target_als = [al for al in self.split_target_alignments + self.gap_sw_realignments
                      if al.reference_name == ti.target
                      and sam.get_strand(al) == ti.sequencing_direction
                      ]
        target_als = sorted(target_als, key=interval.get_covered)

        # Realignment may be have been able to extend an existing alignment.
        target_als = interval.make_parsimonious(target_als)

        merged_als = sam.merge_any_adjacent_pairs(target_als, ti.reference_sequences)

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

    def categorize(self):
        self.outcome = None

        if self.no_alignments_detected:
            self.category = 'bad sequence'
            self.subcategory = 'no alignments detected'
            self.details = 'n/a'
            self.outcome = None

        elif len(self.seq) <= self.target_info.combined_primer_length + 10:
            self.category = 'nonspecific amplification'
            self.subcategory = 'unknown'
            self.details = 'n/a'
            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.single_alignment_covers_read:
            if len(self.indels) == 0:
                if len(self.non_donor_SNVs) == 0:
                    if self.has_donor_SNV:
                        self.category = 'donor'

                        if self.has_nondonor_SNV_at_donor_position:
                            self.subcategory = 'other'
                        else:
                            self.subcategory = 'clean'

                        self.outcome = HDROutcome(self.donor_SNV_string, self.donor_deletions_seen)
                        self.details = str(self.outcome)
                        self.relevant_alignments = [self.single_target_alignment] + self.donor_alignments
                    else:
                        if self.starts_at_expected_location:
                            self.category = 'wild type'
                            self.subcategory = 'clean'
                            self.details = 'n/a'
                        else:
                            edge = self.target_reference_edges['left']

                            if edge is not None:
                                self.category = 'truncation'
                                self.subcategory = 'clean'
                                self.details = str(edge)
                                self.outcome = TruncationOutcome(edge)
                            else:
                                self.category = 'uncategorized'
                                self.subcategory = 'uncategorized'
                                self.details = 'n/a'

                        self.relevant_alignments = [self.single_target_alignment]

                else: # no indels but mismatches
                    if self.has_donor_SNV and not self.error_corrected:
                        self.category = 'donor'
                        if self.has_nondonor_SNV_at_donor_position:
                            self.subcategory = 'other'
                        else:
                            self.subcategory = 'clean'
                        self.outcome = HDROutcome(self.donor_SNV_string, self.donor_deletions_seen)
                        self.details = str(self.outcome)
                        self.relevant_alignments = [self.single_target_alignment] + self.donor_alignments

                    else:
                        if self.starts_at_expected_location:
                            if self.error_corrected:
                                self.category = 'mismatches'
                            else:
                                self.category = 'wild type'

                            self.subcategory = 'mismatches'
                            self.outcome = MismatchOutcome(self.non_donor_SNVs)
                            self.details = str(self.outcome)
                        else:
                            edge = self.target_reference_edges['left']

                            if edge is not None:
                                self.category = 'truncation'
                                self.subcategory = 'mismatches'
                                self.details = str(edge)
                                self.outcome = TruncationOutcome(edge)
                            else:
                                self.category = 'uncategorized'
                                self.subcategory = 'uncategorized'
                                self.details = 'n/a'

                        self.relevant_alignments = [self.single_target_alignment]

            elif len(self.indels) == 1:

                indel, near_cut = self.indels[0]

                if len(self.donor_deletions_seen) == 1:
                    if len(self.non_donor_SNVs) == 0:
                        self.category = 'donor'
                        self.outcome = HDROutcome(self.donor_SNV_string, self.donor_deletions_seen)
                        self.details = str(self.outcome)
                        self.relevant_alignments = [self.single_target_alignment] + self.donor_alignments

                    else: # has non-donor SNVs
                        if self.is_valid_SD_MMEJ:
                            self.register_SD_MMEJ()
                        else:
                            self.category = 'uncategorized'
                            self.subcategory = 'donor deletion plus at least one non-donor mismatch'
                            self.details = 'n/a'
                            self.relevant_alignments = self.uncategorized_relevant_alignments

                else: # one indel, not a donor deletion
                    if self.has_any_SNV:
                        if self.has_donor_SNV and len(self.non_donor_SNVs) == 0:
                            if len(self.indels_besides_one_base_deletions) == 0:
                                # Interpret single base deletions in sequences that otherwise look
                                # like donor incorporation as synthesis errors in oligo donors.
                                self.category = 'donor'
                                self.subcategory = 'synthesis errors'
                                self.outcome = HDROutcome(self.donor_SNV_string, self.donor_deletions_seen)
                                self.details = str(self.outcome)

                                self.relevant_alignments = [self.single_target_alignment] + self.donor_alignments

                                #if len(self.donor_alignments) != 1:
                                #    raise ValueError
                                
                                self.special_alignment = self.donor_alignments[0]

                            else:
                                if indel.kind == 'D':
                                    self.category = 'donor + deletion'
                                    self.subcategory = 'donor + deletion'
                                    HDR_outcome = HDROutcome(self.donor_SNV_string, self.donor_deletions_seen)
                                    deletion_outcome = DeletionOutcome(indel)
                                    self.outcome = HDRPlusDeletionOutcome(HDR_outcome, deletion_outcome)
                                    self.details = str(self.outcome)
                                    self.relevant_alignments = self.target_alignments + self.donor_alignments
                                elif indel.kind == 'I' and indel.length == 1:
                                    self.category = 'donor + insertion'
                                    self.subcategory = 'donor + insertion'
                                    HDR_outcome = HDROutcome(self.donor_SNV_string, self.donor_deletions_seen)
                                    insertion_outcome = InsertionOutcome(indel)
                                    self.outcome = HDRPlusInsertionOutcome(HDR_outcome, insertion_outcome)
                                    self.details = str(self.outcome)
                                    self.relevant_alignments = self.target_alignments + self.donor_alignments
                                else:
                                    # insertion can be a mis-identified duplication of part of HA
                                    if self.donor_insertion is not None:
                                        self.register_donor_insertion()
                                    else:
                                        self.category = 'uncategorized'
                                        self.subcategory = 'donor SNV with non-donor insertion'
                                        self.details = 'n/a'
                                        self.relevant_alignments = self.uncategorized_relevant_alignments

                        elif self.error_corrected and self.is_valid_insertion_with_deletion:
                            self.register_insertion_with_deletion()

                        else:
                            if len(self.non_donor_SNVs) == 1:
                                if indel.kind == 'D':
                                    if near_cut:
                                        if self.error_corrected:
                                            if self.is_valid_insertion_with_deletion:
                                                self.register_insertion_with_deletion()
                                            else:
                                                self.register_deletion_plus_mismatches()
                                        elif self.is_valid_SD_MMEJ and self.is_valid_insertion_with_deletion:
                                            # stop-gap mesaure to prevent mismatches near cut from being ignored
                                            self.register_insertion_with_deletion()
                                        else:
                                            self.register_indel(indel, near_cut)

                                    else: # far from cut
                                        if self.donor_insertion:
                                            self.register_donor_insertion()
                                        else:
                                            self.register_deletion_plus_mismatches()

                                elif indel.kind == 'I':
                                    if self.error_corrected:
                                        self.register_insertion_plus_mismatches()
                                    else:
                                        self.register_indel(indel, near_cut)
                                else:
                                    raise ValueError('bad indel kind', indel.kind)

                            else: # one indel, more than 1 non-donor SNV
                                if self.error_corrected:
                                    if self.is_valid_insertion_with_deletion:
                                        self.register_insertion_with_deletion()
                                    else:
                                        if indel.kind == 'D':
                                            self.register_deletion_plus_mismatches()
                                        elif indel.kind == 'I':
                                            self.register_insertion_plus_mismatches()
                                        else:
                                            raise ValueError('bad indel kind', indel.kind)
                                elif self.is_valid_SD_MMEJ and self.is_valid_insertion_with_deletion:
                                    # stop-gap mesaure to prevent mismatches near cut from being ignored
                                    self.register_insertion_with_deletion()

                                else:
                                    self.register_indel(indel, near_cut)

                    else: # one indel, no SNVs
                        if indel.kind == 'D':
                            if self.non_primer_nts <= 10:
                                self.category = 'nonspecific amplification'
                                self.subcategory = 'primer dimer'
                                self.details = 'n/a'
                                self.relevant_alignments = self.uncategorized_relevant_alignments
                            else:
                                self.register_indel(indel, near_cut)

                        elif indel.kind == 'I':
                            if len(self.indels_near_cut) == 1:
                                self.register_indel(indel, near_cut)
                            else:
                                if indel.length <= 3 and not self.error_corrected:
                                    self.category = 'wild type'
                                    self.subcategory = 'indels'
                                    self.details = 'n/a'
                                
                                else:
                                    self.category = 'uncategorized'
                                    self.subcategory = 'indel far from cut'
                                    self.details = 'n/a'
                                    self.relevant_alignments = self.uncategorized_relevant_alignments

                        self.relevant_alignments = [self.single_target_alignment]

            else: # more than one indel
                if self.has_any_SNV:
                    if self.has_donor_SNV and len(self.non_donor_SNVs) == 0:
                        if len(self.indels_besides_one_base_deletions) == 0:
                            self.category = 'donor'
                            self.subcategory = 'synthesis errors'
                            self.outcome = HDROutcome(self.donor_SNV_string, self.donor_deletions_seen)
                            self.details = str(self.outcome)

                            #if len(self.donor_alignments) != 1:
                            #    raise ValueError
                            
                            self.special_alignment = self.donor_alignments[0]

                        elif self.is_valid_insertion_with_deletion:
                            self.register_insertion_with_deletion()

                        else:
                            if self.donor_insertion is not None:
                                self.register_donor_insertion()
                            else:
                                self.category = 'uncategorized'
                                self.subcategory = 'multiple indels plus at least one donor SNV'
                                self.details = 'n/a'
                                self.relevant_alignments = self.uncategorized_relevant_alignments

                    elif self.is_valid_SD_MMEJ:
                        self.register_SD_MMEJ()

                    else:
                        if self.donor_insertion is not None:
                            self.register_donor_insertion()
                        else:
                            self.category = 'uncategorized'
                            self.subcategory = 'multiple indels plus at least one mismatch'
                            self.details = 'n/a'
                            self.relevant_alignments = self.uncategorized_relevant_alignments

                else: # no SNVs
                    if len(self.donor_deletions_seen) > 0:
                        if len(self.indels_besides_one_base_deletions) == 0:
                            self.category = 'donor'
                            self.subcategory = 'synthesis errors'
                            self.outcome = HDROutcome(self.donor_SNV_string, self.donor_deletions_seen)
                            self.details = str(self.outcome)
                        else:
                            if self.is_valid_SD_MMEJ:
                                self.register_SD_MMEJ()
                            else:
                                self.category = 'uncategorized'
                                self.subcategory = 'multiple indels including donor deletion'
                                self.details = 'n/a'
                                self.relevant_alignments = self.uncategorized_relevant_alignments

                    else:
                        if self.is_valid_insertion_with_deletion:
                            self.register_insertion_with_deletion()
                        else:
                            if set(indel.kind for indel, near_cut in self.indels) == {'D'}:
                                self.category = 'multiple deletions'
                                self.subcategory = 'multiple deletions'
                                deletion_outcomes = sorted([DeletionOutcome(indel) for indel, near_cut in self.indels], key=str)
                                self.outcome = MultipleDeletionOutcome(deletion_outcomes)
                                self.details = str(self.outcome)
                                self.relevant_alignments = self.parsimonious_target_alignments
                            else:
                                self.category = 'uncategorized'
                                self.subcategory = 'multiple indels'
                                self.details = 'n/a'
                                self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.is_valid_insertion_with_deletion:
            self.register_insertion_with_deletion()
        
        elif self.donor_insertion is not None:
            self.register_donor_insertion()
        
        elif self.genomic_insertion is not None:
            self.register_genomic_insertion()

        elif self.duplication is not None:
            subcategory, ref_junctions, indels, als_with_donor_SNVs, merged_als = self.duplication
            if len(indels) == 0:
                self.category = 'duplication'
                self.subcategory = subcategory
                self.outcome = DuplicationOutcome(ref_junctions)
                self.details = str(self.outcome)
                self.relevant_alignments = merged_als
            else:
                self.category = 'uncategorized'
                self.subcategory = 'uncategorized'
                self.details = 'n/a'
                self.relevant_alignments = self.uncategorized_relevant_alignments

        #elif self.multipart_templated_insertion:
        #    self.category = 'complex templated insertion'

        #    best = self.multipart_templated_insertion[0]
        #    sources = set([best['left_source'], best['right_source']])
        #    if sources == set(['genomic']):
        #        self.subcategory = 'genomic'
        #    elif sources == set(['donor']):
        #        self.subcategory = 'donor'
        #    else:
        #        self.subcategory = 'mixed'
        #    
        #    self.details = 'n/a'

        #    self.relevant_alignments = best['full_alignments']

        elif self.phiX_alignments:
            self.category = 'phiX'
            self.subcategory = 'phiX'
            self.details = 'n/a'

        else:
            num_Ns = Counter(self.seq)['N']

            if num_Ns > 10:
                self.category = 'bad sequence'
                self.subcategory = 'low quality'
                self.details = 'n/a'

            elif self.Q30_fractions['all'] < 0.5:
                self.category = 'bad sequence'
                self.subcategory = 'low quality'
                self.details = 'n/a'

            elif self.Q30_fractions['second_half'] < 0.5:
                self.category = 'bad sequence'
                self.subcategory = 'low quality'
                self.details = 'n/a'
                
            elif self.longest_polyG >= 20:
                self.category = 'bad sequence'
                self.subcategory = 'low quality'
                self.details = 'n/a'

            else:
                self.category = 'uncategorized'
                self.subcategory = 'uncategorized'
                self.details = 'n/a'
                
            self.relevant_alignments = self.uncategorized_relevant_alignments

        if self.outcome is not None:
            # Translate positions to be relative to a registered anchor
            # on the target sequence.
            self.details = str(self.outcome.perform_anchor_shift(self.target_info.anchor))

        return self.category, self.subcategory, self.details, self.outcome

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
        return self.target_alignments + self.donor_alignments + interval.make_parsimonious(self.nonredundant_supplemental_alignments) + self.gap_sw_realignments
