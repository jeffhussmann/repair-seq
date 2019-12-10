import warnings
import multiprocessing

import pandas as pd
import Bio.SeqIO
import tqdm

from knock_knock import target_info
from hits import fasta, mapping_tools, utilities
from ddr import guide_library

def build_guide_specific_target(original_target, guide_library, guide_row):
    print(guide_library, guide_row.name)

    guide = guide_row.name
    
    warnings.simplefilter('ignore')

    new_name = f'pooled_vector_{guide_library}_{guide}'

    new_dir = original_target.dir.parent / new_name
    new_dir.mkdir(exist_ok=True)

    ps = original_target.features['pooled_vector', 'protospacer']

    gb = Bio.SeqIO.read('/home/jah/projects/ddr/targets/pooled_vector/pooled_vector.gb', 'genbank')

    gb.seq = gb.seq[:ps.start] + guide_row['protospacer'] + gb.seq[ps.end + 1:]

    Bio.SeqIO.write(gb, str(new_dir / 'pooled_vector.gb'), 'genbank')

    for fn in ['phiX.gb', 'donors.gb', 'manifest.yaml']:
        try:
            (new_dir / fn).symlink_to(original_target.dir / fn)
        except FileExistsError:
            pass

    new_ti = target_info.TargetInfo(original_target.base_dir, new_name)

    new_ti.make_references()
    new_ti.identify_degenerate_indels()

def build_doubles_guide_specific_target(original_target,
                                        fixed_guide_library,
                                        variable_guide_library,
                                        fixed_guide,
                                        variable_guide,
                                        tasks_queue=None,
                                       ):
    warnings.simplefilter('ignore')

    new_name = f'{original_target.name}-{fixed_guide}-{variable_guide}'

    new_dir = original_target.dir.parent / new_name
    new_dir.mkdir(exist_ok=True)


    gb_fn = original_target.dir / f'{original_target.name}.gb'
    gb = Bio.SeqIO.read(str(gb_fn), 'genbank')

    fixed_ps = original_target.features[original_target.name, 'fixed_protospacer']
    fixed_ps_seq = fixed_guide_library.guides_df.loc[fixed_guide, 'protospacer']
    gb.seq = gb.seq[:fixed_ps.start] + fixed_ps_seq + gb.seq[fixed_ps.end + 1:]

    variable_ps = original_target.features[original_target.name, 'variable_protospacer']
    variable_ps_seq = variable_guide_library.guides_df.loc[variable_guide, 'protospacer']
    gb.seq = gb.seq[:variable_ps.start] + variable_ps_seq + gb.seq[variable_ps.end + 1:]

    guide_bc_start = original_target.features[original_target.name, 'fixed_guide_barcode'].start
    guide_bc_end = original_target.features[original_target.name, 'sequencing_start'].end
    fixed_bc_seq_rc = fixed_guide_library.guides_df.loc[fixed_guide, 'guide_barcode']
    fixed_bc_seq = utilities.reverse_complement(fixed_bc_seq_rc)
    gb.seq = gb.seq[:guide_bc_start] + fixed_bc_seq + gb.seq[guide_bc_end + 1:]

    Bio.SeqIO.write(gb, str(new_dir / f'{original_target.name}.gb'), 'genbank')

    for fn in ['phiX.gb', 'donors.gb', 'manifest.yaml']:
        try:
            (new_dir / fn).symlink_to(original_target.dir / fn)
        except FileExistsError:
            pass

    new_ti = target_info.TargetInfo(original_target.base_dir, new_name)

    new_ti.make_references()
    new_ti.identify_degenerate_indels()

    if tasks_queue is not None:
        tasks_queue.put((fixed_guide, variable_guide))

#if __name__ == '__main__':
#    base_dir = '/home/jah/projects/ddr'
#    original_target = target_info.TargetInfo(base_dir, 'pooled_vector')
#
#    original_target.make_references()
#    original_target.identify_degenerate_indels()
#
#    args = []
#
#    for guide_library in [
#        'DDR_library',
#        'DDR_sublibrary',
#    ]:
#        guides = pd.read_table(f'/home/jah/projects/ddr/guides/{guide_library}/guides.txt', index_col='short_name')
#        for guide in guides.index:
#            args.append((original_target, guide_library, guides.loc[guide]))
#
#    pool = multiprocessing.Pool(processes=18)
#    pool.starmap(build_guide_specific_target, args)

if __name__ == '__main__':
    warnings.simplefilter('ignore')

    base_dir = '/home/jah/projects/ddr'
    original_target = target_info.TargetInfo(base_dir, 'doubles_vector')

    original_target.make_references()
    original_target.identify_degenerate_indels()

    args_list = []

    fixed_guide_library = guide_library.GuideLibrary(base_dir, 'DDR_skinny')
    variable_guide_library = guide_library.GuideLibrary(base_dir, 'DDR_sublibrary')

    manager = multiprocessing.Manager()
    tasks_done_queue = manager.Queue()

    for fixed_guide in fixed_guide_library.guides:
        for variable_guide in variable_guide_library.guides:
            args = (original_target, fixed_guide_library, variable_guide_library, fixed_guide, variable_guide, tasks_done_queue)
            args_list.append(args)

    #args_list = args_list[:10]
    progress = tqdm.tqdm(desc='Making doubles_vector targets', total=len(args_list))
    pool = multiprocessing.Pool(processes=18)
    pool.starmap_async(build_doubles_guide_specific_target, args_list)

    while progress.n != len(args_list):
        message = tasks_done_queue.get()
        progress.update()

    #for args in tqdm.tqdm(args_list[:10]):
    #    build_doubles_guide_specific_target(*args)