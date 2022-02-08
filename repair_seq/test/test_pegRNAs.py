from pathlib import Path

import knock_knock.pegRNAs
import knock_knock.target_info

base_dir = Path(__file__).parent

def test_twin_prime_intended_deletion_inferrence():
    DegenerateDeletion = knock_knock.target_info.DegenerateDeletion

    for pegRNA_pair, expected in [
        (('sample11_pegRNA1', 'sample11_pegRNA2'), DegenerateDeletion([3223, 3224, 3225, 3226], 30)),
        (('sample12_pegRNA1', 'sample12_pegRNA2'), DegenerateDeletion([3204], 30)),
        (('sample13_pegRNA1', 'sample13_pegRNA2'), DegenerateDeletion([3215, 3216], 28)),
        (('sample14_pegRNA1', 'sample14_pegRNA2'), DegenerateDeletion([3212], 34)),
    ]:
        ti = knock_knock.target_info.TargetInfo(base_dir, 'pPC1655', pegRNAs=pegRNA_pair, sequencing_start_feature_name='forward_primer')
        deletion, deletion_feature = knock_knock.pegRNAs.infer_twin_pegRNA_intended_deletion(ti)
        assert (deletion == expected)