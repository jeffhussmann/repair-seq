'''
Extracting alignments for manually identified read+expected outcome pairs
and regression testing these categorizations.
'''

import shutil
import textwrap
from pathlib import Path

import yaml

import hits.utilities
import hits.sam
import knock_knock.target_info

import repair_seq as rs

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

    def get_read_layout(self, read_id):
        layout = None

        for qname, als in hits.sam.grouped_by_name(self.bam_fn):
            if qname == read_id:
                layout = self.categorizer(als, self.target_info)
                break

        if layout is None:
            raise ValueError(read_id)
        
        return layout

    @memoized_property
    def target_info(self):
        target_info_name = self.details['target_info']
        supplemental_index_names = ['hg19', 'bosTau7', 'e_coli']
        supplemental_indices = knock_knock.target_info.locate_supplemental_indices(base_dir)
        supplemental_indices = {name: supplemental_indices[name] for name in supplemental_index_names}
        target_info = knock_knock.target_info.TargetInfo(base_dir,
                                                         target_info_name,
                                                         supplemental_indices=supplemental_indices,
                                                         **self.details.get('target_info_kwargs', {}),
                                                        )

        return target_info

    @memoized_property
    def categorizer(self):
        if self.details['experiment_type'] == 'twin_prime':
            from repair_seq.twin_prime_layout import Layout
            categorizer = Layout
        elif self.details['experiment_type'] in ('prime_editing', 'prime_editing_layout'):
            from repair_seq.prime_editing_layout import Layout
            categorizer = Layout
        else:
            raise NotImplementedError

        return categorizer

def build_all_pooled_screen_read_sets(only_new=False):
    src_read_sets_dir = base_dir / 'manual_read_sets' / 'pooled_screens'
    read_set_fns = src_read_sets_dir.glob('*.yaml')

    for read_set_fn in read_set_fns:
        set_name = read_set_fn.stem

        read_set = ReadSet(set_name)

        if read_set.dir.is_dir() and only_new:
            continue
        else:
            build_pooled_screen_read_set(set_name)

def build_pooled_screen_read_set(set_name):
    read_set_fn = base_dir / 'manual_read_sets' / 'pooled_screens' / f'{set_name}.yaml'
    manual_details = yaml.safe_load(read_set_fn.read_text())

    read_set = ReadSet(set_name)

    read_set.dir.mkdir(exist_ok=True, parents=True)
    
    pool = rs.pooled_screen.get_pool(manual_details['base_dir'], manual_details['pool_name'])
    exp = pool.single_guide_experiment(manual_details['fixed_guide'], manual_details['variable_guide'])

    # Note: might prefer to have exp-specific target_info, but these have symlinks that would need
    # to be resolved. Instead, just use the pool target_info.
    new_target_info_dir = base_dir / 'targets' / pool.target_info.name
    existing_target_info_dir = pool.target_info.dir

    if new_target_info_dir.is_dir():
        approved_deletion = input(f'Deleting target directory {new_target_info_dir}, proceed?') == 'y'
        if approved_deletion:
            shutil.rmtree(new_target_info_dir)

    shutil.copytree(existing_target_info_dir, new_target_info_dir)

    # Pools may specify specialized values for some target_info
    # parameters that need to be passed along.
    possible_target_info_kwargs_keys = [
        'pegRNAs',
        'sequencing_start_feature_name',
        'primer_names',
    ]

    target_info_kwargs = {
        key: exp.description[key]
        for key in possible_target_info_kwargs_keys
        if key in exp.description
    }

    alignment_sorter = hits.sam.AlignmentSorter(read_set.bam_fn, exp.combined_header, by_name=True)

    read_info = {
        'experiment_type': pool.sample_sheet['layout_module'],
        'target_info': pool.target_info.name,
        'target_info_kwargs': target_info_kwargs,
        'expected_values': {},
    }

    with alignment_sorter:
        for read_id, details in manual_details['expected_values'].items():
            als = exp.get_read_alignments(read_id)
            for al in als:
                # Overwrite potential common sequence query_name. 
                al.query_name = read_id
                alignment_sorter.write(al)
                
            read_info['expected_values'][read_id] = details

    read_set.expected_values_fn.write_text(yaml.safe_dump(read_info, sort_keys=False))
        
def build_all_arrayed_group_read_sets(only_new=False):
    src_read_sets_dir = base_dir / 'manual_read_sets' / 'arrayed_groups'
    read_set_fns = src_read_sets_dir.glob('*.yaml')

    for read_set_fn in read_set_fns:
        set_name = read_set_fn.stem

        read_set = ReadSet(set_name)

        if read_set.dir.is_dir() and only_new:
            continue
        else:
            build_arrayed_group_read_set(set_name)

def build_arrayed_group_read_set(set_name):
    read_set_fn = base_dir / 'manual_read_sets' / 'arrayed_groups' / f'{set_name}.yaml'
    manual_details = yaml.safe_load(read_set_fn.read_text())

    read_set = ReadSet(set_name)

    read_set.dir.mkdir(exist_ok=True, parents=True)

    exps = rs.arrayed_experiment_group.get_all_experiments(manual_details['base_dir'])
    exp = exps[manual_details['batch_name'], manual_details['group_name'], manual_details['exp_name']]
    exp_type = exp.experiment_group.description['experiment_type']

    # Experiments may specify specialized values for some target_info
    # parameters that need to be passed along.
    possible_target_info_kwargs_keys = [
        'pegRNAs',
        'sequencing_start_feature_name',
        'primer_names',
    ]

    target_info_kwargs = {
        key: exp.description[key]
        for key in possible_target_info_kwargs_keys
        if key in exp.description
    }

    new_target_info_dir = base_dir / 'targets' / exp.target_info.name
    existing_target_info_dir = exp.target_info.dir

    if new_target_info_dir.is_dir():
        approved_deletion = input(f'Deleting target directory {new_target_info_dir}, proceed?') == 'y'
        if approved_deletion:
            shutil.rmtree(new_target_info_dir)

    shutil.copytree(existing_target_info_dir, new_target_info_dir)

    alignment_sorter = hits.sam.AlignmentSorter(read_set.bam_fn, exp.combined_header, by_name=True)
    
    read_info = {
        'experiment_type': exp_type,
        'target_info': exp.target_info.name,
        'target_info_kwargs': target_info_kwargs,
        'expected_values': {},
    }

    with alignment_sorter:
        for read_id, details in manual_details['expected_values'].items():
            als = exp.get_read_alignments(read_id)
            for al in als:
                # Overwrite potential common sequence query_name. 
                al.query_name = read_id
                alignment_sorter.write(al)
                
            read_info['expected_values'][read_id] = details

    read_set.expected_values_fn.write_text(yaml.safe_dump(read_info, sort_keys=False))

def test_read_sets():
    read_set_dirs = sorted([p for p in (base_dir / 'read_sets').iterdir() if p.is_dir()])

    # Ensure that at least one read set was found. 
    assert len(read_set_dirs) > 0

    discrepancies = []

    for read_set_dir in read_set_dirs:
        set_name = read_set_dir.name

        read_set = ReadSet(set_name)

        num_tested = 0

        for qname, als in hits.sam.grouped_by_name(read_set.bam_fn):
            try:
                layout = read_set.categorizer(als, read_set.target_info)
                layout.categorize()

                category = layout.category
                subcategory = layout.subcategory

            except:
                category = 'error'
                subcategory = 'error'

            expected = read_set.expected_values[qname]

            if expected['category'] != category or expected['subcategory'] != subcategory:
                discrepancies.append((set_name, qname, expected, category, subcategory))

            num_tested += 1

        print(f'Tested {num_tested: >3d} sequences for {set_name}.') 

    diagnostic_messages = []
    for set_name, qname, expected, category, subcategory in discrepancies:

        diagnostic_message = f'''
            set name: {set_name}
            query name: {qname}
            expected: ({expected["category"]}, {expected["subcategory"]})
            actual: ({category}, {subcategory})
            note: {expected["note"]}
        '''

        diagnostic_messages.append(textwrap.dedent(diagnostic_message))
    
    full_diagnostic_message = '\n'.join(diagnostic_messages)
    assert len(discrepancies) == 0, full_diagnostic_message