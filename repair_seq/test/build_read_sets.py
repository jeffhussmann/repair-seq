'''
Extracts alignments for manually identified read+expected outcome pairs
for regression testing purposes.
'''

import shutil
from pathlib import Path

import yaml

import hits.sam

import repair_seq.pooled_screen
import repair_seq.arrayed_experiment_group
from repair_seq.test.test_read_sets import ReadSet, base_dir

dsb_pools = repair_seq.pooled_screen.get_all_pools('/lab/solexa_weissman/jah/projects/ddr')

dsb_manual_read_sets = [
]

def build_pooled_screen_read_sets():
    src_read_sets_dir = base_dir / 'manual_read_sets' / 'pooled_screens'
    read_set_fns = src_read_sets_dir.glob('*.yaml')

    for read_set_fn in read_set_fns:
        set_name = read_set_fn.stem

        manual_details = yaml.safe_load(read_set_fn.read_text())

        read_set = ReadSet(set_name)
        read_set.dir.mkdir(exist_ok=True, parents=True)
        
        pool = repair_seq.pooled_screen.get_pool(manual_details['base_dir'], manual_details['pool_name'])
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

        header = hits.sam.get_header(exp.fns_by_read_type['bam_by_name']['collapsed_uncommon_R2'])
        
        alignment_sorter = hits.sam.AlignmentSorter(read_set.bam_fn, header, by_name=True)

        read_info = {
            'experiment_type': pool.sample_sheet['layout_module'],
            'target_info': pool.target_info.name,
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
        
def build_arrayed_group_read_sets():
    src_read_sets_dir = base_dir / 'manual_read_sets' / 'arrayed_groups'
    read_set_fns = src_read_sets_dir.glob('*.yaml')

    for read_set_fn in read_set_fns:
        set_name = read_set_fn.stem

        manual_details = yaml.safe_load(read_set_fn.read_text())

        read_set = ReadSet(set_name)
        read_set.dir.mkdir(exist_ok=True, parents=True)

        exps = repair_seq.arrayed_experiment_group.get_all_experiments(manual_details['base_dir'])
        exp = exps[manual_details['batch_name'], manual_details['exp_name']]
        exp_type = exp.experiment_group.description['experiment_type']

        new_target_info_dir = base_dir / 'targets' / exp.target_info.name
        existing_target_info_dir = exp.target_info.dir

        if new_target_info_dir.is_dir():
            approved_deletion = input(f'Deleting target directory {new_target_info_dir}, proceed?') == 'y'
            if approved_deletion:
                shutil.rmtree(new_target_info_dir)

        shutil.copytree(existing_target_info_dir, new_target_info_dir)

        header = hits.sam.get_header(exp.fns_by_read_type['bam_by_name']['nonredundant'])
        alignment_sorter = hits.sam.AlignmentSorter(read_set.bam_fn, header, by_name=True)
        
        read_info = {
            'experiment_type': exp_type,
            'target_info': exp.target_info.name,
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