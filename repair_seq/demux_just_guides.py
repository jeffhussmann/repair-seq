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
import tqdm
import yaml

from hits import fastq, utilities
import knock_knock.target_info

import repair_seq.annotations
import repair_seq.demux_gDNA
import repair_seq.guide_library
import repair_seq.pooled_screen

def load_sample_sheet(base_dir, batch):
    sample_sheet_fn = Path(base_dir) / 'data' / batch / 'gDNA_sample_sheet.yaml'
    sample_sheet = yaml.safe_load(sample_sheet_fn.read_text())

    pool_details = repair_seq.demux_gDNA.load_pool_details(base_dir, batch)

    global_values = {
        'pooled': True,
        'gDNA': True,
        'R1_primer': sample_sheet.get('R1_primer'),
        'R2_primer': sample_sheet.get('R2_primer'),
        'variable_guide_library': sample_sheet.get('variable_guide_library'),
        'has_UMIs': False,
        'categorizer': sample_sheet.get('categorizer', 'pooled_layout'),
        'R1_read_lengths': (43, 45),
        'target_info_prefix': sample_sheet.get('target_info_prefix', 'pooled_vector'),
    }

    for sample_name, details in pool_details.items():
        sample_sheet['samples'][sample_name].update(global_values)
        sample_sheet['samples'][sample_name].update(details)

    return sample_sheet

def make_pool_sample_sheets(base_dir, batch):
    sample_sheet = load_sample_sheet(base_dir, batch)

    for pool_name, details in sample_sheet['samples'].items():
        full_pool_name = f'{batch}_{pool_name}'
        pool_sample_sheet_fn = Path(base_dir) / 'results' / full_pool_name / 'sample_sheet.yaml'
        pool_sample_sheet_fn.parent.mkdir(parents=True, exist_ok=True)
        pool_sample_sheet_fn.write_text(yaml.safe_dump(details, default_flow_style=False))

    return sample_sheet['samples']

def get_chunks_dir(base_dir, batch, sample_name):
    group_dir = base_dir / 'data' / batch
    
    chunks_dir = group_dir / f'{sample_name}_chunks'

    return chunks_dir

class FastqChunker:
    def __init__(self, base_dir, batch, sample_name, which, reads_per_chunk=None, queue=None, debug=False):
        self.base_dir = Path(base_dir)
        self.batch = batch
        self.sample_name = sample_name
        self.which = which
        self.reads_per_chunk = reads_per_chunk
        self.queue = queue
        self.debug = debug
        
        sample_details = load_sample_sheet(self.base_dir, self.batch)['samples'][sample_name]
        
        self.fn_name = sample_details[self.which]
        
        group_dir = self.base_dir / 'data' / self.batch
        self.input_fn = (group_dir / self.fn_name).with_suffix('.fastq.gz')
        
        self.chunks_dir = get_chunks_dir(self.base_dir, self.batch, self.sample_name)
        
        self.current_chunk_number = 0
        
        self.current_fh = None
        
    def close_current_chunk(self):
        if self.current_fh is not None:
            self.current_fh.close()
            if self.queue is not None:
                self.queue.put(('chunk', self.which, self.current_chunk_number))
                
            self.current_chunk_number += 1
            
    def get_chunk_fn(self, chunk_number):
        chunk_string = repair_seq.demux_gDNA.chunk_number_to_string(chunk_number)
        next_fn = self.chunks_dir / f'{self.fn_name}_{chunk_string}.fastq.gz'
        return next_fn
            
    def start_next_chunk(self):
        self.close_current_chunk()
        
        chunk_fn = self.get_chunk_fn(self.current_chunk_number)
        self.current_fh = gzip.open(chunk_fn, 'wt', compresslevel=1)
        
    def split_into_chunks(self):
        self.chunks_dir.mkdir(exist_ok=True)

        line_groups = fastq.get_line_groups(self.input_fn)

        if self.debug:
            line_groups = itertools.islice(line_groups, int(5e6))
        
        for read_number, line_group in enumerate(line_groups):
            if read_number % self.reads_per_chunk == 0:
                self.start_next_chunk()
            
            for line in line_group:
                self.current_fh.write(line)
                
        self.close_current_chunk()
        
        self.queue.put(('chunk', self.which, 'DONE'))

class Writers:
    def __init__(self, base_dir, batch, sample_name, chunk_number):
        self.base_dir = Path(base_dir)
        
        self.batch = batch
        
        self.chunk_string = f'{sample_name}_{repair_seq.demux_gDNA.chunk_number_to_string(chunk_number)}'

        self.writers = defaultdict(list)

        pool_name = f'{self.batch}_{sample_name}'
        self.pool = repair_seq.pooled_screen.PooledScreenNoUMI(self.base_dir, pool_name)

    def __getitem__(self, key):
        return self.writers[key]

    def write(self):
        for variable_guide in sorted(self.writers):
            reads = self.writers[variable_guide]
            sorted_reads = sorted(reads, key=lambda r: r.name)

            exp = self.pool.single_guide_experiment('none', variable_guide)
            output_dir = exp.fns['chunks']
            output_dir.mkdir(exist_ok=True, parents=True)
            
            fn = output_dir / f'{self.chunk_string}_R2.fastq.gz'
            
            with gzip.open(fn, 'wt', compresslevel=1) as zfh:
                for read in sorted_reads:
                    zfh.write(str(read))

            del self.writers[variable_guide]
            del sorted_reads
                
def split_into_chunks(base_dir, batch, sample_name, which, reads_per_chunk, queue, debug):
    chunker = FastqChunker(base_dir, batch, sample_name, which, reads_per_chunk, queue, debug)
    return chunker.split_into_chunks()

def get_resolvers(base_dir, batch, sample_name):
    sample_details = load_sample_sheet(base_dir, batch)['samples'][sample_name]

    expected_seqs = {}
    resolvers = {}

    variable_guide_library = repair_seq.guide_library.GuideLibrary(base_dir, sample_details['variable_guide_library'])

    ti_prefix = sample_details['target_info_prefix']

    guide_seqs = {}
    guide_seqs['variable_guide'] = defaultdict(list)

    guide_pairs = [('none', vg) for vg in variable_guide_library.guides]

    for fg, vg in guide_pairs:
        ti_name = f'{ti_prefix}_{variable_guide_library.name}_{vg}'
        
        ti = knock_knock.target_info.TargetInfo(base_dir, ti_name)
        
        R1_primer = ti.features[ti.target, sample_details['R1_primer']]
        # Implicit assumption about sequencing direction here.
        
        target_seq = ti.reference_sequences[ti.target]
        
        for R1_read_length in sample_details['R1_read_lengths']:
            expected_R1 = target_seq[R1_primer.start:R1_primer.start + R1_read_length]
            guide_seqs['variable_guide'][vg].append(expected_R1)
        
    # convert from defaultdict to dict
    guide_seqs['variable_guide'] = dict(guide_seqs['variable_guide'])

    resolvers['variable_guide'] = utilities.get_one_mismatch_resolver(guide_seqs['variable_guide']).get
    expected_seqs['variable_guide'] = set()
    for seqs in guide_seqs['variable_guide'].values():
        for seq in seqs:
            expected_seqs['variable_guide'].add(seq)

    return resolvers, expected_seqs

def demux_chunk(base_dir, batch, sample_name, chunk_number, queue):
    resolvers, expected_seqs = get_resolvers(base_dir, batch, sample_name)
    
    fastq_fns = {which: FastqChunker(base_dir, batch, sample_name, which).get_chunk_fn(chunk_number) for which in ['R1', 'R2']}

    writers = Writers(base_dir, batch, sample_name, chunk_number)

    counts = defaultdict(Counter)

    Annotation = repair_seq.annotations.Annotations['R2_with_guide']

    for R1, R2 in fastq.read_pairs(fastq_fns['R1'], fastq_fns['R2'], standardize_names=True):
        if R1.name != R2.name:
            raise ValueError('read pair out of sync')

        variable_guides = resolvers['variable_guide'](R1.seq, {'unknown'})
        if len(variable_guides) == 1:
            variable_guide = next(iter(variable_guides))
        else:
            variable_guide = 'unknown'

        counts['variable_guide'][R1.seq] += 1
        counts['id'][variable_guide] += 1

        if variable_guide == 'unknown':
            continue

        query_name = R1.name
        guide_seq = R1.seq
        guide_qual = fastq.sanitize_qual(R1.qual)

        annotation = Annotation(query_name=query_name,
                                guide=guide_seq,
                                guide_qual=guide_qual,
                               )
        R2.name = str(annotation)

        writers[variable_guide].append(R2)

    writers.write()

    chunks_dir = get_chunks_dir(base_dir, batch, sample_name)
    for key, cs in counts.items():
        cs = pd.Series(cs).sort_values(ascending=False)
        fn = chunks_dir / f'{key}_{repair_seq.demux_gDNA.chunk_number_to_string(chunk_number)}.txt'
        cs.to_csv(fn, sep='\t', header=False)

    # Delete the chunk.
    for fastq_fn in fastq_fns.values():
        fastq_fn.unlink()

    queue.put(('demux', sample_name, chunk_number))

def merge_seq_counts(base_dir, batch, sample_name, key):
    chunks_dir = get_chunks_dir(base_dir, batch, sample_name)
    # Note: glob returns generator, need to make into list
    # so it can be re-used for cleanup.
    count_fns = sorted(chunks_dir.glob(f'{key}*.txt'))

    counts = Counter()
    for fn in count_fns:
        for line in open(fn):
            seq, count = line.strip().split()
            counts[seq] += int(count)

    merged_fn = Path(base_dir) / 'data' / batch / f'{sample_name}_{key}_stats.txt'

    resolvers, expected_seqs, *_ = get_resolvers(base_dir, batch, sample_name)
    resolver = resolvers[key]
    expected_seqs = expected_seqs[key]

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

def merge_ids(base_dir, batch, sample_name):
    chunks_dir = get_chunks_dir(base_dir, batch, sample_name)
    # Note: glob returns generator, need to make into list
    # so it can be re-used for cleanup.
    count_fns = sorted(chunks_dir.glob('id*.txt'))

    counts = Counter()
    for fn in count_fns:
        for line in open(fn):
            variable_guide, count = line.strip().split()
            counts[variable_guide] += int(count)

    merged_fn = Path(base_dir) / 'data' / batch / f'{sample_name}_id_stats.txt'

    counts = pd.Series(counts).sort_index()
    counts.to_csv(merged_fn, sep='\t', header=False)

    for fn in count_fns:
        fn.unlink()

    return counts
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', type=Path, default=Path.home() / 'projects' / 'repair_seq')
    parser.add_argument('batch')
    parser.add_argument('sample_name')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    base_dir = args.base_dir
    batch = args.batch
    debug = args.debug
    sample_name = args.sample_name
    
    sample_details = load_sample_sheet(base_dir, batch)['samples'][sample_name]
    
    reads_per_chunk = int(5e6)

    total_chunks = int(np.ceil(sample_details['num_reads'] / reads_per_chunk))

    manager = multiprocessing.Manager()
    tasks_done_queue = manager.Queue()
    
    chunks_done = defaultdict(set) 
    
    chunk_progress = tqdm.tqdm(desc='Chunk progress', total=total_chunks)
    demux_progress = tqdm.tqdm(desc='Demux progress', total=total_chunks)
    
    chunk_pool = multiprocessing.Pool(processes=2)
    demux_pool = multiprocessing.Pool(processes=4)

    with chunk_pool, demux_pool:
        chunk_results = []
        demux_results = []

        unfinished_chunkers = set()
        
        for which in ['R1', 'R2']:
            args = (base_dir, batch, sample_name, which, reads_per_chunk, tasks_done_queue, debug)
            chunk_result = chunk_pool.apply_async(split_into_chunks, args)
            if debug:
                print(chunk_result.get())
            chunk_results.append(chunk_result)

            unfinished_chunkers.add(which)
            
        chunk_pool.close()
        
        while True:
            task_type, which, chunk_number = tasks_done_queue.get()
            
            if task_type == 'chunk':
                
                if chunk_number == 'DONE':
                    unfinished_chunkers.remove(which)
                    if len(unfinished_chunkers) == 0:
                        break
                else:
                    chunks_done[chunk_number].add(which)
                    if chunks_done[chunk_number] == set(['R1', 'R2']):
                        chunk_progress.update()
                        
                        args = (base_dir, batch, sample_name, chunk_number, tasks_done_queue)
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

    merge_seq_counts(base_dir, batch, sample_name, 'variable_guide')
    counts = merge_ids(base_dir, batch, sample_name)

    chunks_dir = get_chunks_dir(base_dir, batch, sample_name)
    shutil.rmtree(chunks_dir)

    full_pool_name = f'{batch}_{sample_name}'
    pool = repair_seq.pooled_screen.PooledScreenNoUMI(base_dir, full_pool_name)

    counts.index = pd.MultiIndex.from_tuples([('none', guide) for guide in counts.index], names=['fixed_guide', 'variable_guide'])
    counts.name = 'num_reads'
    counts = counts.sort_values(ascending=False)
    counts.to_csv(pool.fns['read_counts'], sep='\t', header=True)