import warnings
import multiprocessing
import argparse

import pandas as pd
import Bio.SeqIO
import tqdm

from knock_knock import target_info
from hits import fasta, mapping_tools, utilities
from ddr.guide_library import GuideLibrary

def build_guide_specific_target(original_target,
                                original_genbank_name,
                                guide_library,
                                guide,
                                tasks_queue=None,
                               ):
    warnings.simplefilter('ignore')

    new_name = f'{original_target.name}_{guide_library.name}_{guide}'

    new_dir = original_target.dir.parent / new_name
    new_dir.mkdir(exist_ok=True)

    gb_fn = original_target.dir / f'{original_genbank_name}.gb'
    gb = Bio.SeqIO.read(str(gb_fn), 'genbank')

    protospacer = original_target.features[original_target.target, 'protospacer']
    ps_seq = guide_library.guides_df.loc[guide, 'protospacer']
    gb.seq = gb.seq[:protospacer.start] + ps_seq + gb.seq[protospacer.end + 1:]

    Bio.SeqIO.write(gb, str(new_dir / f'{original_genbank_name}.gb'), 'genbank')

    fns_to_copy = [f'{source}.gb' for source in original_target.sources]
    fns_to_copy.append('manifest.yaml')
    for fn in fns_to_copy:
        try:
            (new_dir / fn).symlink_to(original_target.dir / fn)
        except FileExistsError:
            pass

    new_ti = target_info.TargetInfo(original_target.base_dir, new_name)

    new_ti.make_references()
    new_ti.identify_degenerate_indels()

    if tasks_queue is not None:
        tasks_queue.put(guide)

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

def build_all_singles(base_dir, original_target_name, original_genbank_name, guide_library_names):
    warnings.simplefilter('ignore')

    original_target = target_info.TargetInfo(base_dir, original_target_name)

    original_target.make_references()
    original_target.identify_degenerate_indels()

    args_list = []

    manager = multiprocessing.Manager()
    tasks_done_queue = manager.Queue()

    for guide_library_name in guide_library_names:
        guide_library = GuideLibrary(base_dir, guide_library_name)
        for guide in guide_library.guides:
            args_list.append((original_target, original_genbank_name, guide_library, guide, tasks_done_queue))

    progress = tqdm.tqdm(desc='Making targets', total=len(args_list))

    #args_list = args_list[:10]
    #for args in args_list:
    #    build_guide_specific_target(*args)

    with multiprocessing.Pool(processes=18) as pool:
        pool.starmap_async(build_guide_specific_target, args_list)

        while progress.n != len(args_list):
            tasks_done_queue.get()
            progress.update()

def build_all_doubles():
    warnings.simplefilter('ignore')

    base_dir = '/home/jah/projects/ddr'
    original_target = target_info.TargetInfo(base_dir, 'doubles_vector')

    original_target.make_references()
    original_target.identify_degenerate_indels()

    args_list = []

    fixed_guide_library = GuideLibrary(base_dir, 'DDR_skinny')
    variable_guide_library = GuideLibrary(base_dir, 'DDR_sublibrary')

    manager = multiprocessing.Manager()
    tasks_done_queue = manager.Queue()

    for fixed_guide in fixed_guide_library.guides:
        for variable_guide in variable_guide_library.guides:
            args = (original_target, fixed_guide_library, variable_guide_library, fixed_guide, variable_guide, tasks_done_queue)
            args_list.append(args)

    #args_list = args_list[:10]
    #for args in tqdm.tqdm(args_list):
    #    build_doubles_guide_specific_target(*args)

    progress = tqdm.tqdm(desc='Making doubles_vector targets', total=len(args_list))
    pool = multiprocessing.Pool(processes=18)
    pool.starmap_async(build_doubles_guide_specific_target, args_list)

    while progress.n != len(args_list):
        tasks_done_queue.get()
        progress.update()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vector')

    args = parser.parse_args()

    #if args.vector == 'singles':
    #    base_dir = '/home/jah/projects/prime_editing_screens'
    #    original_target = 'pPC1000_G6C_15'
    #    original_genbank_name = 'ppc1000'
    #    guide_library_names = ['DDR_library']
    #    build_all_singles(base_dir, original_target, original_genbank_name, guide_library_names)
    if args.vector == 'singles':
        base_dir = '/home/jah/projects/ddr'
        original_target = 'pooled_vector'
        original_genbank_name = 'pooled_vector'
        guide_library_names = ['DDR_library', 'DDR_sublibrary']
        build_all_singles(base_dir, original_target, original_genbank_name, guide_library_names)
    elif args.vector == 'doubles':
        build_all_doubles()
