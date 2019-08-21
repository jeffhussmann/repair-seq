import warnings
import multiprocessing

import pandas as pd
import Bio.SeqIO

from knock_knock import target_info
from hits import fasta, mapping_tools, utilities

def build_guide_specific_target(original_target, guide_library, guide_row):
    print(guide_library, guide_row.name)

    guide = guide_row.name
    
    warnings.simplefilter('ignore')

    new_name = f'pooled_vector_{guide_library}_{guide}'

    new_dir = original.dir.parent / new_name
    new_dir.mkdir(exist_ok=True)

    ps = original.features['pooled_vector', 'protospacer']

    gb = Bio.SeqIO.read('/home/jah/projects/ddr/targets/pooled_vector/pooled_vector.gb', 'genbank')

    gb.seq = gb.seq[:ps.start] + guide_row['protospacer'] + gb.seq[ps.end + 1:]

    Bio.SeqIO.write(gb, str(new_dir / 'pooled_vector.gb'), 'genbank')

    for fn in ['phiX.gb', 'donors.gb', 'manifest.yaml']:
        try:
            (new_dir / fn).symlink_to(original.dir / fn)
        except FileExistsError:
            pass

    new_ti = target_info.TargetInfo(base_dir, new_name)

    new_ti.make_references()
    new_ti.identify_degenerate_indels()

if __name__ == '__main__':
    original_target = target_info.TargetInfo(base_dir, 'pooled_vector')

    original_target.make_references()
    original_target.identify_degenerate_indels()

    args = []

    for guide_library in [
        'DDR_library',
        'DDR_sublibrary',
    ]:
        guides = pd.read_table(f'/home/jah/projects/ddr/guides/{guide_library}/guides.txt', index_col='short_name')
        for guide in guides.index:
            args.append((original_target, guide_library, guides.loc[guide]))

    pool = multiprocessing.Pool(processes=18)
    pool.starmap(build_guide_specific_target, args)