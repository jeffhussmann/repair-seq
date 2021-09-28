from collections import Counter

from knock_knock import outcome_record
from hits.utilities import group_by

from .collapse_cython import hamming_distance_matrix, register_corrections

Pooled_UMI_Outcome = outcome_record.OutcomeRecord_factory(
    columns_arg=[
        'UMI',
        'guide_mismatch',
        'cluster_id',
        'num_reads',
        'inferred_amplicon_length',
        'category',
        'subcategory',
        'details',
        'query_name',
        'common_sequence_name',
    ],
    converters_arg={
        'num_reads': int,
        'guide_mismatch': int,
        'inferred_amplicon_length': int,
    },
)

gDNA_Outcome = outcome_record.OutcomeRecord_factory(
    columns_arg=[
        'query_name',
        'guide_mismatches',
        'inferred_amplicon_length',
        'category',
        'subcategory',
        'details',
        'common_sequence_name',
    ],
    converters_arg={'inferred_amplicon_length': int},
)

def collapse_pooled_UMI_outcomes(outcome_iter):
    def is_relevant(outcome):
        return (outcome.category != 'bad sequence' and
                outcome.outcome != ('no indel', 'other', 'ambiguous')
               )

    all_outcomes = [o for o in outcome_iter if is_relevant(o)]
    all_outcomes = sorted(all_outcomes, key=lambda u: (u.UMI, u.cluster_id))

    all_collapsed_outcomes = []
    most_abundant_outcomes = []

    for UMI, UMI_outcomes in group_by(all_outcomes, lambda u: u.UMI):
        observed = set(u.outcome for u in UMI_outcomes)

        collapsed_outcomes = []
        for outcome in observed:
            relevant = [u for u in UMI_outcomes if u.outcome == outcome]
            representative = max(relevant, key=lambda u: u.num_reads)
            representative.num_reads = sum(u.num_reads for u in relevant)

            collapsed_outcomes.append(representative)
            all_collapsed_outcomes.append(representative)
    
        max_count = max(u.num_reads for u in collapsed_outcomes)
        has_max_count = [u for u in collapsed_outcomes if u.num_reads == max_count]

        if len(has_max_count) == 1:
            most_abundant_outcomes.append(has_max_count[0])

    all_collapsed_outcomes = sorted(all_collapsed_outcomes, key=lambda u: (u.UMI, u.cluster_id))
    return all_collapsed_outcomes, most_abundant_outcomes
        
def error_correct_outcome_UMIs(outcome_group, max_UMI_distance=1):
    # sort UMIs in descending order by number of occurrences.
    UMI_read_counts = Counter()
    for outcome in outcome_group:
        UMI_read_counts[outcome.UMI] += outcome.num_reads
    UMIs = [UMI for UMI, read_count in UMI_read_counts.most_common()]

    ds = hamming_distance_matrix(UMIs)

    corrections = register_corrections(ds, max_UMI_distance, UMIs)

    for outcome in outcome_group:
        correct_to = corrections.get(outcome.UMI)
        if correct_to:
            outcome.UMI = correct_to
    
    return outcome_group