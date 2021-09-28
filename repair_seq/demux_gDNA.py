import argparse
import gzip
import multiprocessing
import itertools
import shutil

from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
idx = pd.IndexSlice
import yaml
import tqdm

from hits import fastq, utilities
import knock_knock.target_info
import repair_seq.guide_library
import repair_seq.pooled_screen
import repair_seq.guide_library
import repair_seq.collapse

def load_sample_sheet(base_dir, batch):
    sample_sheet_fn = Path(base_dir) / 'data' / batch / 'gDNA_sample_sheet.yaml'
    sample_sheet = yaml.safe_load(sample_sheet_fn.read_text())

    pool_details = load_pool_details(base_dir, batch)
    if pool_details is not None:
        sample_sheet['pool_details'] = pool_details

    data_dir = base_dir / 'data' / batch 

    R1_fns = [(data_dir / d['R1']).with_suffix('.fastq.gz') for d in sample_sheet['quartets'].values()]
    R1_lengths = {fastq.get_read_length(R1_fn) for R1_fn in R1_fns}
    if len(R1_lengths) != 1:
        raise ValueError(R1_lengths)
    else:
        R1_length = R1_lengths.pop()

    sample_sheet['R1_read_length'] = R1_length

    return sample_sheet

def load_pool_details(base_dir, batch):
    pool_details_fn = Path(base_dir) / 'data' / batch / 'gDNA_pool_details.csv'
    if pool_details_fn.exists():
        pool_details = pd.read_csv(pool_details_fn, index_col='pool_name').replace({np.nan: None})

        for key_to_split in ['I7_index', 'I5_index', 'sgRNA']:
            if key_to_split in pool_details:
                pool_details[key_to_split] = [s.split(';') if s is not None else s for s in pool_details[key_to_split]]
        pool_details = pool_details.T.to_dict()
    else:
        pool_details = None
    return pool_details

def make_pool_sample_sheets(base_dir, batch):
    sample_sheet = load_sample_sheet(base_dir, batch)

    for pool_name, details in sample_sheet['pool_details'].items():
        pool_sample_sheet = {
            'pooled': True,
            'gDNA': True,
            'R1_primer': sample_sheet['R1_primer'],
            'R2_primer': sample_sheet['R2_primer'],
            'variable_guide_library': sample_sheet['variable_guide_library'],
            'has_UMIs': sample_sheet.get('has_UMIs', False),
            'layout_module': sample_sheet['layout_module'],
            'infer_homology_arms': sample_sheet.get('infer_homology_arms', True),
        }

        if 'target_info_prefix' in sample_sheet:
            target_info_prefix = sample_sheet['target_info_prefix']
        else:
            target_info_prefix = 'pooled_vector'

        if 'fixed_guide_library' in sample_sheet:
            pool_sample_sheet['fixed_guide_library'] = sample_sheet['fixed_guide_library']

        pool_sample_sheet['target_info_prefix'] = target_info_prefix

        pool_sample_sheet.update(details)

        full_pool_name = f'{sample_sheet["group_name"]}_{pool_name}'
        pool_sample_sheet_fn = Path(base_dir) / 'results' / full_pool_name / 'sample_sheet.yaml'
        pool_sample_sheet_fn.parent.mkdir(parents=True, exist_ok=True)
        pool_sample_sheet_fn.write_text(yaml.safe_dump(pool_sample_sheet, default_flow_style=False))

    return sample_sheet['pool_details']

def chunk_number_to_string(chunk_number):
    return f'{chunk_number:06d}'

class FastqChunker:
    def __init__(self, base_dir, batch, quartet_name, which, reads_per_chunk=None, queue=None, debug=False):
        self.base_dir = Path(base_dir)
        self.batch = batch
        self.quartet_name = quartet_name
        self.which = which
        self.reads_per_chunk = reads_per_chunk
        self.queue = queue
        self.debug = debug
        
        sample_sheet = load_sample_sheet(self.base_dir, self.batch)
        quartet = sample_sheet['quartets'][quartet_name]
        
        self.fn_name = quartet[self.which]
        
        group_dir = self.base_dir / 'data' / self.batch
        self.input_fn = (group_dir / self.fn_name).with_suffix('.fastq.gz')
        
        self.chunks_dir = group_dir / 'gDNA_chunks'
        self.chunks_dir.mkdir(exist_ok=True)
        
        self.current_chunk_number = 0
        
        self.current_fh = None
        
    def close_current_chunk(self):
        if self.current_fh is not None:
            self.current_fh.close()
            if self.queue is not None:
                self.queue.put(('chunk', self.quartet_name, self.which, self.current_chunk_number))
                
            self.current_chunk_number += 1
            
    def get_chunk_fn(self, chunk_number):
        chunk_string = chunk_number_to_string(chunk_number)
        next_fn = self.chunks_dir / f'{self.fn_name}_{chunk_string}.fastq.gz'
        return next_fn
            
    def start_next_chunk(self):
        self.close_current_chunk()
        
        chunk_fn = self.get_chunk_fn(self.current_chunk_number)
        self.current_fh = gzip.open(chunk_fn, 'wt', compresslevel=1)
        
    def split_into_chunks(self):
        line_groups = fastq.get_line_groups(self.input_fn)

        if self.debug:
            line_groups = itertools.islice(line_groups, int(5e6))
        
        for read_number, line_group in enumerate(line_groups):
            if read_number % self.reads_per_chunk == 0:
                self.start_next_chunk()
            
            for line in line_group:
                self.current_fh.write(line)
                
        self.close_current_chunk()
        
        self.queue.put(('chunk', self.quartet_name, self.which, 'DONE'))

class Writers:
    def __init__(self, base_dir, batch, quartet_name, chunk_number):
        self.base_dir = Path(base_dir)
        
        sample_sheet = load_sample_sheet(base_dir, batch)
        self.batch = sample_sheet['group_name']
        
        self.chunk_string = f'{quartet_name}_{chunk_number_to_string(chunk_number)}'

        self.writers = defaultdict(list)

    def __getitem__(self, key):
        return self.writers[key]

    def write(self):
        pools = {}

        for sample, fixed_guide, variable_guide in sorted(self.writers):
            if sample not in pools:
                pool_name = f'{self.batch}_{sample}'
                pool = repair_seq.pooled_screen.PooledScreenNoUMI(self.base_dir, pool_name)
                pools[sample] = pool

            pool = pools[sample]

            reads = self.writers[sample, fixed_guide, variable_guide]
            sorted_reads = sorted(reads, key=lambda r: r.name)

            exp = pool.single_guide_experiment(fixed_guide, variable_guide)
            output_dir = exp.fns['chunks']
            output_dir.mkdir(exist_ok=True, parents=True)
            
            fn = output_dir / f'{self.chunk_string}_R2.fastq.gz'
            
            with gzip.open(fn, 'wt', compresslevel=1) as zfh:
                for read in sorted_reads:
                    zfh.write(str(read))

            del self.writers[sample, fixed_guide, variable_guide]
            del sorted_reads
                
def split_into_chunks(base_dir, group, quartet_name, which, reads_per_chunk, queue, debug):
    chunker = FastqChunker(base_dir, group, quartet_name, which, reads_per_chunk, queue, debug)
    return chunker.split_into_chunks()

def get_resolvers(base_dir, group):
    sample_sheet = load_sample_sheet(base_dir, group)

    expected_seqs = {}
    resolvers = {}

    I7_indices = {name: details['I7_index'] for name, details in sample_sheet['pool_details'].items()}
    I5_indices = {name: details['I5_index'] for name, details in sample_sheet['pool_details'].items()}

    expected_I7_indices = set()
    for seqs in I7_indices.values():
        if not isinstance(seqs, list):
            seqs = [seqs]
        expected_I7_indices.update(seqs)

    expected_I5_indices = set()
    for seqs in I5_indices.values():
        if not isinstance(seqs, list):
            seqs = [seqs]
        expected_I5_indices.update(seqs)

    expected_seqs['I7'] = expected_I7_indices
    expected_seqs['I5'] = expected_I5_indices
    resolvers['I7'] = utilities.get_one_mismatch_resolver(I7_indices).get
    resolvers['I5'] = utilities.get_one_mismatch_resolver(I5_indices).get

    variable_guide_library = repair_seq.guide_library.GuideLibrary(base_dir, sample_sheet['variable_guide_library'])

    ti_prefix = sample_sheet['target_info_prefix']

    guide_seqs = {}
    guide_seqs['variable_guide'] = defaultdict(set)

    if 'fixed_guide_library' in sample_sheet:
        has_fixed_barcode = True
        fixed_guide_library = repair_seq.guide_library.GuideLibrary(base_dir, sample_sheet['fixed_guide_library'])
        guide_seqs['fixed_guide_barcode'] = defaultdict(set)
        guide_pairs = list(itertools.product(fixed_guide_library.guides, variable_guide_library.guides))
    else:
        has_fixed_barcode = False
        guide_pairs = [('none', vg) for vg in variable_guide_library.guides]

    for fg, vg in guide_pairs:
        if fg == 'none':
            ti_name = f'{ti_prefix}_{variable_guide_library.name}_{vg}'
        else:
            ti_name = f'{ti_prefix}-{fg}-{vg}'
        
        ti = knock_knock.target_info.TargetInfo(base_dir, ti_name)
        
        R1_primer = ti.features[ti.target, sample_sheet['R1_primer']]
        R2_primer = ti.features[ti.target, sample_sheet['R2_primer']]
        
        target_seq = ti.reference_sequences[ti.target]
        
        expected_R1 = target_seq[R1_primer.start:R1_primer.start + sample_sheet['R1_read_length']]
        guide_seqs['variable_guide'][vg].add(expected_R1)
        
        if fg != 'none':
            fixed_guide_barcode = ti.features[ti.target, 'fixed_guide_barcode']

            expected_R2 = utilities.reverse_complement(target_seq[fixed_guide_barcode.start:R2_primer.end + 1])
            guide_seqs['fixed_guide_barcode'][fg].add(expected_R2)
        
    for which in ['fixed_guide_barcode', 'variable_guide']:
        if which in guide_seqs:
            dictionary = guide_seqs[which]
            
            for g in sorted(dictionary):
                seqs = dictionary[g]
                if len(seqs) != 1:
                    raise ValueError(which, g, seqs)
                else:
                    seq = seqs.pop()
                    dictionary[g] = seq
                    
            # convert from defaultdict to dict
            guide_seqs[which] = dict(dictionary)

    if has_fixed_barcode:
        fixed_lengths = {len(s) for s in guide_seqs['fixed_guide_barcode'].values()}
        if len(fixed_lengths) != 1:
            raise ValueError(fixed_lengths)

        fixed_length = fixed_lengths.pop()
        guide_barcode_slice = idx[:fixed_length]
        # If a guide barcode is present, remove it from R2 before passing along
        # to simplify analysis of common sequences in pool.
        after_guide_barcode_slice = idx[fixed_length:]

        resolvers['fixed_guide_barcode'] = utilities.get_one_mismatch_resolver(guide_seqs['fixed_guide_barcode']).get
        expected_seqs['fixed_guide_barcode'] = set(guide_seqs['fixed_guide_barcode'].values())
    else:
        # If there weren't multiple fixed guide pools present, keep everything
        # to allow possibility of outcomes that don't include the intended NotI site.
        def fixed_guide_barcode_resolver(*args):
            return {'none'}

        resolvers['fixed_guide_barcode'] = fixed_guide_barcode_resolver
        expected_seqs['fixed_guide_barcode'] = set()

        guide_barcode_slice = slice(None)
        after_guide_barcode_slice = idx[:]

    resolvers['variable_guide'] = utilities.get_one_mismatch_resolver(guide_seqs['variable_guide']).get
    expected_seqs['variable_guide'] = set(guide_seqs['variable_guide'].values())

    return resolvers, expected_seqs, guide_barcode_slice, after_guide_barcode_slice

def demux_chunk(base_dir, group, quartet_name, chunk_number, queue):
    resolvers, expected_seqs, guide_barcode_slice, after_guide_barcode_slice = get_resolvers(base_dir, group)
    
    fastq_fns = [FastqChunker(base_dir, group, quartet_name, which).get_chunk_fn(chunk_number) for which in fastq.quartet_order]

    writers = Writers(base_dir, group, quartet_name, chunk_number)

    counts = defaultdict(Counter)

    Annotation = repair_seq.collapse.Annotations['R2_with_guide']

    for quartet in fastq.read_quartets(fastq_fns, standardize_names=True):
        if len({r.name for r in quartet}) != 1:
            raise ValueError('quartet out of sync')

        I7_samples = resolvers['I7'](quartet.I1.seq, {'unknown'})
        I5_samples = resolvers['I5'](quartet.I2.seq, {'unknown'})

        consistent_with_both = I7_samples & I5_samples

        if len(consistent_with_both) == 1:
            sample = next(iter(consistent_with_both))
        else:
            sample = 'unknown'

        counts['I7'][quartet.I1.seq] += 1
        counts['I5'][quartet.I2.seq] += 1

        # Note: primer for gDNA prep makes R1 read start 1 downstream of UMI prep.
        variable_guide_seq = quartet.R1.seq

        variable_guides = resolvers['variable_guide'](variable_guide_seq, {'unknown'})
        if len(variable_guides) == 1:
            variable_guide = next(iter(variable_guides))
        else:
            variable_guide = 'unknown'

        counts['variable_guide'][variable_guide_seq] += 1

        guide_barcode = quartet.R2.seq[guide_barcode_slice]
        fixed_guides = resolvers['fixed_guide_barcode'](guide_barcode, {'unknown'})
        if len(fixed_guides) == 1:
            fixed_guide = next(iter(fixed_guides))
        else:
            fixed_guide = 'unknown'

        counts['fixed_guide_barcode'][guide_barcode] += 1

        counts['id'][sample, fixed_guide, variable_guide] += 1

        # Retain quartets with an unknown fixed guide to allow detection of weird ligations.
        if 'unknown' in {sample, variable_guide}:
            continue

        query_name = quartet.R1.name
        guide_seq = quartet.R1.seq
        guide_qual = fastq.sanitize_qual(quartet.R1.qual)

        annotation = Annotation(query_name=query_name,
                                guide=guide_seq,
                                guide_qual=guide_qual,
                               )
        quartet.R2.name = str(annotation)

        writers[sample, fixed_guide, variable_guide].append(quartet.R2[after_guide_barcode_slice])

    writers.write()

    chunk_dir = base_dir / 'data' / group / 'gDNA_chunks'
    for k, cs in counts.items():
        cs = pd.Series(cs).sort_values(ascending=False)
        fn = chunk_dir / f'{k}_{quartet_name}_{chunk_number_to_string(chunk_number)}.txt'
        cs.to_csv(fn, sep='\t', header=False)

    # Delete the chunk.
    for fastq_fn in fastq_fns:
        fastq_fn.unlink()

    queue.put(('demux', quartet_name, chunk_number))

def merge_seq_counts(base_dir, group, k):
    chunk_dir = Path(base_dir) / 'data' / group / 'gDNA_chunks'
    # Note: glob returns generator, need to make into list
    # so it can be re-used for cleanup.
    count_fns = sorted(chunk_dir.glob(f'{k}*.txt'))

    counts = Counter()
    for fn in count_fns:
        for line in open(fn):
            seq, count = line.strip().split()
            counts[seq] += int(count)

    merged_fn = Path(base_dir) / 'data' / group / f'{k}_stats.txt'

    resolvers, expected_seqs, *_ = get_resolvers(base_dir, group)
    resolver = resolvers[k]
    expected_seqs = expected_seqs[k]

    with open(merged_fn, 'w') as fh:
        total = sum(counts.values())

        for seq, count in counts.most_common(100):
            name = resolver(seq, '')

            if isinstance(name, set):
                name = sorted(name)

            if seq in expected_seqs:
                mismatches = ''
            elif name != '':
                mismatches = ' (1 mismatch)'
            else:
                mismatches = ''

            fraction = float(count) / total

            fh.write(f'{seq}\t{count: >10,}\t({fraction: >6.2%})\t{name}{mismatches}\n')

    for fn in count_fns:
        fn.unlink()
    
def merge_ids(base_dir, group):
    chunk_dir = Path(base_dir) / 'data' / group / 'gDNA_chunks'
    # Note: glob returns generator, need to make into list
    # so it can be re-used for cleanup.
    count_fns = sorted(chunk_dir.glob('id*.txt'))

    counts = Counter()
    for fn in count_fns:
        for line in open(fn):
            sample, fixed_guide, variable_guide, count = line.strip().split()
            counts[sample, fixed_guide, variable_guide] += int(count)

    merged_fn = Path(base_dir) / 'data' / group / 'id_stats.txt'

    counts = pd.Series(counts).sort_index()
    counts.to_csv(merged_fn, sep='\t', header=False)

    for fn in count_fns:
        fn.unlink()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', type=Path, default=Path.home() / 'projects' / 'repair_seq')
    parser.add_argument('batch')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    base_dir = args.base_dir
    batch = args.batch
    debug = args.debug
    
    pool_details = make_pool_sample_sheets(base_dir, batch)

    sample_sheet = load_sample_sheet(base_dir, batch)
    
    quartet_names = sorted(sample_sheet['quartets'])
    reads_per_chunk = int(5e6)

    chunks_per_quartet = [int(np.ceil(d['num_reads'] / reads_per_chunk)) for q, d in sample_sheet['quartets'].items()]
    total_chunks = sum(chunks_per_quartet)

    manager = multiprocessing.Manager()
    tasks_done_queue = manager.Queue()
    
    chunks_done = defaultdict(set) 
    
    chunk_progress = tqdm.tqdm(desc='Chunk progress', total=total_chunks)
    demux_progress = tqdm.tqdm(desc='Demux progress', total=total_chunks)
    
    chunk_pool = multiprocessing.Pool(processes=4 * len(quartet_names))
    demux_pool = multiprocessing.Pool(processes=4 * len(quartet_names))

    with chunk_pool, demux_pool:
        chunk_results = []
        demux_results = []

        unfinished_chunkers = set()
        
        for quartet_name in quartet_names:
            for which in fastq.quartet_order:
                args = (base_dir, batch, quartet_name, which, reads_per_chunk, tasks_done_queue, debug)
                chunk_result = chunk_pool.apply_async(split_into_chunks, args)
                if debug:
                    print(chunk_result.get())
                chunk_results.append(chunk_result)

                unfinished_chunkers.add((quartet_name, which))
            
        chunk_pool.close()
        
        while True:
            task_type, *task_info = tasks_done_queue.get()
            
            if task_type == 'chunk':
                quartet_name, which, chunk_number = task_info
                
                if chunk_number == 'DONE':
                    unfinished_chunkers.remove((quartet_name, which))
                    if len(unfinished_chunkers) == 0:
                        break
                else:
                    chunks_done[quartet_name, chunk_number].add(which)
                    if chunks_done[quartet_name, chunk_number] == set(fastq.quartet_order):
                        chunk_progress.update()
                        
                        args = (base_dir, batch, quartet_name, chunk_number, tasks_done_queue)
                        demux_result = demux_pool.apply_async(demux_chunk, args)
                        if debug:
                            print(demux_result.get())
                        demux_results.append(demux_result)

            elif task_type == 'demux':
                demux_progress.update()

        while demux_progress.n < chunk_progress.n:
            task_type, *task_info = tasks_done_queue.get()
            if task_type == 'demux':
                demux_progress.update()
            else:
                error, = task_info
                raise error
        
        chunk_pool.join()
        for chunk_result in chunk_results:
            if not chunk_result.successful():
                print(chunk_result.get())
                
        demux_pool.close()
                
        demux_pool.join()
        for demux_result in demux_results:
            if not demux_result.successful():
                print(demux_result.get())

    chunk_progress.close()
    demux_progress.close()

    print('Merging statistics...')

    merge_pool = multiprocessing.Pool(processes=3)
    merge_results = []
    merge_results.append(merge_pool.apply_async(merge_seq_counts, args=(base_dir, batch, 'I5')))
    merge_results.append(merge_pool.apply_async(merge_seq_counts, args=(base_dir, batch, 'I7')))
    merge_results.append(merge_pool.apply_async(merge_seq_counts, args=(base_dir, batch, 'variable_guide')))
    merge_results.append(merge_pool.apply_async(merge_seq_counts, args=(base_dir, batch, 'fixed_guide_barcode')))
    merge_results.append(merge_pool.apply_async(merge_ids, args=(base_dir, batch)))

    merge_pool.close()
    merge_pool.join()

    for merge_result in merge_results:
        if not merge_result.successful():
            print(merge_result.get())

    chunk_dir = base_dir / 'data' / batch / 'gDNA_chunks'
    shutil.rmtree(str(chunk_dir))

    merged_fn = Path(base_dir) / 'data' / batch / 'id_stats.txt'
    id_counts = pd.read_csv(merged_fn, sep='\t', header=None, names=['pool', 'fixed_guide', 'variable_guide', 'num_reads'], index_col=[0, 1, 2], squeeze=True)

    for pool_name in pool_details:
        if pool_name in id_counts:
            full_pool_name = f"{sample_sheet['group_name']}_{pool_name}"
            pool = repair_seq.pooled_screen.PooledScreen(base_dir, full_pool_name)
            counts = id_counts.loc[pool_name].sort_values(ascending=False)
            counts.to_csv(pool.fns['read_counts'], sep='\t', header=True)