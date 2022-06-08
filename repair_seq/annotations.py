import hits.annotation

annotation_fields = {
    'UMI': [
        ('UMI', 's'),
        ('original_name', 's'),
    ],

    'UMI_guide': [
        ('UMI', 's'),
        ('guide', 's'),
        ('guide_qual', 's'),
        ('original_name', 's'),
    ],

    'collapsed_UMI': [
        ('UMI', 's'),
        ('guide', 's'),
        ('guide_qual', 's'),
        ('cluster_id', '06d'),
        ('num_reads', '06d'),
    ],

    'collapsed_UMI_mismatch': [
        ('UMI', 's'),
        ('cluster_id', '06d'),
        ('num_reads', '010d'),
        ('mismatch', 'd'),
    ],

    'common_sequence': [
        ('rank', '012d'),
        ('count', '012d'),
    ],

    'R2_with_guide': [
        ('query_name', 's'),
        ('guide', 's'),
        ('guide_qual', 's'),
    ],

    'R2_with_guide_mismatches': [
        ('query_name', 's'),
        ('mismatches', 's'),
    ],

    'SRA': [
        ('original_name', 's'),
        ('UMI_seq', 's'),
        ('UMI_qual', 's'),
    ],
}

Annotations = {key: hits.annotation.Annotation_factory(fields) for key, fields in annotation_fields.items()}