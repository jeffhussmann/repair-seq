from pathlib import Path

import hits.sam
import knock_knock.target_info
import repair_seq.prime_editing_layout

base_dir = Path.home() / 'projects' / 'repair_seq' / 'code' / 'repair_seq' / 'test')

class ReadSet:
    def __init__(self, set_name, target_info_name):
        self.set_name = set_name
        self.target_info_name = target_info_name
        
        supplemental_index_names = ['hg19', 'bosTau7', 'e_coli']
        supplemental_indices = knock_knock.target_info.locate_supplemental_indices(base_dir)
        supplemental_indices = {name: supplemental_indices[name] for name in supplemental_index_names}
        self.target_info = knock_knock.target_info.TargetInfo(base_dir, target_info_name, supplemental_indices=supplemental_indices)
        
        self.bam_fn = base_dir / f'alignments_for_{self.set_name}.bam'
        self.expected_values_fn = base_dir / f'expected_values_for_{set_name}.txt'
        
    def expected_values(self):
        expected_values = {}
        with open(self.expected_values_fn) as expected_values_fh:
            for line in expected_values_fh:
                read_id, category, subcategory, description = line.strip().split('\t')
                expected_values[read_id] = {
                    'category': category,
                    'subcategory': subcategory,
                    'description': description,
                }
        return expected_values

def test_read_sets():
    with open(base_dir / 'read_set_details.txt') as set_details_fh:
        for set_details_line in set_details_fh:
            set_name, target_info_name = set_details_line.strip().split('\t')
            read_set = ReadSet(set_name, target_info_name)

            actual_values = {}
            for qname, als in hits.sam.grouped_by_name(read_set.bam_fn):
                l = repair_seq.prime_editing_layout.Layout(als, read_set.target_info)
                l.categorize()
                actual_values[qname] = {
                    'category': l.category,
                    'subcategory': l.subcategory,
                }

            for qname, expected in read_set.expected_values().items():
                actual = actual_values[qname]
                assert (expected['category'], expected['subcategory']) == (actual['category'], actual['subcategory']), f'{set_name} {expected["description"]}'