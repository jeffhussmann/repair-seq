from pathlib import Path

import knock_knock.pegRNAs
import knock_knock.target_info

base_dir = Path(__file__).parent

def test_twin_prime_intended_deletion_inferrence():
    DegenerateDeletion = knock_knock.target_info.DegenerateDeletion

    for pegRNA_pair, expected, is_prime_del in [
        (('sample01_pegRNA1', 'sample01_pegRNA2'), DegenerateDeletion([3203, 3204], 50), True),
        (('sample11_pegRNA1', 'sample11_pegRNA2'), DegenerateDeletion([3223, 3224, 3225, 3226], 30), False),
        (('sample12_pegRNA1', 'sample12_pegRNA2'), DegenerateDeletion([3204], 30), False),
        (('sample13_pegRNA1', 'sample13_pegRNA2'), DegenerateDeletion([3215, 3216], 28), False),
        (('sample14_pegRNA1', 'sample14_pegRNA2'), DegenerateDeletion([3212], 34), False),
        (('220224_sample07_pegRNA1', '220224_sample07_pegRNA2'), None, False), # point mutation, not a deletion
    ]:

        # Test in both forward and reverse orientations:
        for sequencing_start_feature_name in ['forward_primer', 'gDNA_reverse_primer']:
            ti = knock_knock.target_info.TargetInfo(base_dir,
                                                    'pPC1655',
                                                    pegRNAs=pegRNA_pair,
                                                    sequencing_start_feature_name=sequencing_start_feature_name,
                                                   )

            assert (ti.twin_pegRNA_intended_deletion == expected)
            assert (ti.is_prime_del == is_prime_del)

def test_pegRNA_feature_inferrence():
    ti = knock_knock.target_info.TargetInfo(base_dir, 'PAH_E4-2_45_EvoPreQ1-4_43_EvoPreQ1')
    feature = ti.features['PAH_E4', 'HA_PBS_PAH_E4.2_45_EvoPreQ1']
    assert feature.start == 612
    assert feature.end == 624
    assert feature.strand == '-'

    feature = ti.features['PAH_E4', 'HA_PBS_PAH_E4.4_43_EvoPreQ1'] 
    assert feature.start == 536
    assert feature.end ==  547
    assert feature.strand == '+'

def test_twin_prime_overlap_inferrence():
    ti = knock_knock.target_info.TargetInfo(base_dir, 'HEK3_attB_A30_B30')

    A_feature = ti.features['HEK3_attB_A_30', 'overlap']
    B_feature = ti.features['HEK3_attB_B_30', 'overlap']
    assert A_feature.start == 96
    assert A_feature.end == 117

    assert B_feature.start == 96
    assert B_feature.end == 117

    assert {A_feature.strand, B_feature.strand} == {'+', '-'}

    ti = knock_knock.target_info.TargetInfo(base_dir, 'PAH_E4-2_45_EvoPreQ1-4_43_EvoPreQ1')

    feature_2_45 = ti.features['PAH_E4.2_45_EvoPreQ1', 'overlap']
    assert feature_2_45.start == 97
    assert feature_2_45.end == 120

    feature_4_43 = ti.features['PAH_E4.4_43_EvoPreQ1', 'overlap']
    assert feature_4_43.start == 97
    assert feature_4_43.end == 120

    assert {feature_2_45.strand, feature_4_43.strand} == {'+', '-'}