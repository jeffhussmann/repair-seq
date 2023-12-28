import knock_knock.outcome_record

UMI_Outcome = knock_knock.outcome_record.OutcomeRecord_factory(
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

Outcome = knock_knock.outcome_record.OutcomeRecord_factory(
    columns_arg=[
        'query_name',
        'guide_mismatches',
        'inferred_amplicon_length',
        'category',
        'subcategory',
        'details',
        'common_sequence_name',
    ],
    converters_arg={
        'inferred_amplicon_length': int,
    },
)