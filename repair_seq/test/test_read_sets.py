from pathlib import Path

import yaml

import hits.utilities
import hits.sam
import knock_knock.target_info

memoized_property = hits.utilities.memoized_property

base_dir = Path(__file__).parent

class ReadSet:
    def __init__(self, set_name):
        self.set_name = set_name

        self.dir = base_dir / 'read_sets' / self.set_name
        
        self.bam_fn =  self.dir / 'alignments.bam'
        self.expected_values_fn = self.dir / 'expected_values.yaml'

    @memoized_property
    def details(self):
        return yaml.safe_load(self.expected_values_fn.read_text())

    @memoized_property
    def expected_values(self):
        return self.details['expected_values']

    @memoized_property
    def target_info(self):
        target_info_name = self.details['target_info']
        supplemental_index_names = ['hg19', 'bosTau7', 'e_coli']
        supplemental_indices = knock_knock.target_info.locate_supplemental_indices(base_dir)
        supplemental_indices = {name: supplemental_indices[name] for name in supplemental_index_names}
        target_info = knock_knock.target_info.TargetInfo(base_dir, target_info_name, supplemental_indices=supplemental_indices)

        if self.details['experiment_type'] == 'twin_prime':
            target_info.infer_pegRNA_features()
            target_info.infer_twin_pegRNA_overlap()

        return target_info

    @memoized_property
    def categorizer(self):
        if self.details['experiment_type'] == 'twin_prime':
            from repair_seq.twin_prime_layout import Layout
            categorizer = Layout
        elif self.details['experiment_type'] == 'prime_editing_layout':
            from repair_seq.prime_editing_layout import Layout
            categorizer = Layout
        else:
            raise NotImplementedError

        return categorizer
    
    #def expected_values(self):
    #    expected_values = {}
    #    with open(self.expected_values_fn) as expected_values_fh:
    #        for line in expected_values_fh:
    #            read_id, category, subcategory, description = line.strip().split('\t')
    #            expected_values[read_id] = {
    #                'category': category,
    #                'subcategory': subcategory,
    #                'description': description,
    #            }
    #    return expected_values

#def test_read_sets():
#    with open(base_dir / 'read_set_details.txt') as set_details_fh:
#        for set_details_line in set_details_fh:
#            set_name, target_info_name = set_details_line.strip().split('\t')
#            read_set = ReadSet(set_name, target_info_name)
#
#            actual_values = {}
#            for qname, als in hits.sam.grouped_by_name(read_set.bam_fn):
#                l = repair_seq.prime_editing_layout.Layout(als, read_set.target_info)
#                l.categorize()
#                actual_values[qname] = {
#                    'category': l.category,
#                    'subcategory': l.subcategory,
#                }
#
#            for qname, expected in read_set.expected_values().items():
#                actual = actual_values[qname]
#                assert (expected['category'], expected['subcategory']) == (actual['category'], actual['subcategory']), f'{set_name} {expected["description"]}'

def test_read_sets():
    read_set_dirs = [p for p in (base_dir / 'read_sets').iterdir() if p.is_dir()]

    # Ensure that at least one read set was found. 
    assert len(read_set_dirs) > 0

    for read_set_dir in read_set_dirs:
        print(read_set_dir)
        set_name = read_set_dir.name

        read_set = ReadSet(set_name)

        actual_values = {}
        for qname, als in hits.sam.grouped_by_name(read_set.bam_fn):
            l = read_set.categorizer(als, read_set.target_info)
            l.categorize()
            actual_values[qname] = {
                'category': l.category,
                'subcategory': l.subcategory,
            }

        for qname, expected in read_set.expected_values.items():
            actual = actual_values[qname]
            assert (expected['category'], expected['subcategory']) == (actual['category'], actual['subcategory']), f'{read_set.set_name} {qname} {expected["note"]}'