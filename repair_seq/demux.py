import argparse
import gzip
import itertools
import logging
import multiprocessing
import shutil

from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
idx = pd.IndexSlice
import yaml
import tqdm

from hits import fastq, fasta, utilities

import repair_seq.guide_library
from repair_seq.annotations import Annotations

def load_sample_sheet(base_dir, group):
    sample_sheet_fn = Path(base_dir) / 'data' / group / 'sample_sheet.yaml'
    sample_sheet = yaml.safe_load(sample_sheet_fn.read_text())
    pool_details = load_pool_details(base_dir, group)
    if pool_details is not None:
        sample_sheet['pool_details'] = pool_details
    return sample_sheet

def load_pool_details(base_dir, group):
    pool_details_fn = Path(base_dir) / 'data' / group / 'pool_details.csv'
    if pool_details_fn.exists():
        pool_details = pd.read_csv(pool_details_fn, index_col='pool_name').replace({np.nan: None})
        pool_details['index'] = [s.split(';') for s in pool_details['index']]
        pool_details = pool_details.T.to_dict()
    else:
        pool_details = None
    return pool_details

def make_pool_sample_sheets(base_dir, group):
    sample_sheet = load_sample_sheet(base_dir, group)

    for pool_name, details in sample_sheet['pool_details'].items():
        pool_sample_sheet = {
            'pooled': True,
            'variable_guide_library': sample_sheet['variable_guide_library'],
        }

        if 'fixed_guide_library' in sample_sheet:
            pool_sample_sheet['fixed_guide_library'] = sample_sheet['fixed_guide_library']
            pool_sample_sheet['target_info_prefix'] = 'doubles_vector'
        else:
            pool_sample_sheet['target_info_prefix'] = 'pooled_vector'

        pool_sample_sheet.update(details)

        full_pool_name = f'{sample_sheet["group_name"]}_{pool_name}'
        pool_sample_sheet_fn = Path(base_dir) / 'results' / full_pool_name / 'sample_sheet.yaml'
        pool_sample_sheet_fn.parent.mkdir(parents=True, exist_ok=True)
        pool_sample_sheet_fn.write_text(yaml.safe_dump(pool_sample_sheet, default_flow_style=False))

def load_SRA_sample_sheet():
    sample_sheet_fn = Path(__file__).parent / 'metadata' / 'SRA_sample_sheet.csv'
    SRA_sample_sheet = pd.read_csv(sample_sheet_fn, index_col='screen_name')
    return SRA_sample_sheet

def load_SRR_accessions():
    sample_sheet_fn = Path(__file__).parent / 'metadata' / 'SRR_accessions.csv'
    SRR_accessions = pd.read_csv(sample_sheet_fn, index_col=['screen_name', 'SRR_accession'])
    return SRR_accessions

def load_SRA_pool_sample_sheet(screen_name):
    SRA_sample_sheet = load_SRA_sample_sheet()
    SRR_accessions = load_SRR_accessions()

    row = SRA_sample_sheet.loc[screen_name]
    row = row.replace([np.nan], [None])
    sample_sheet = row.to_dict()

    sample_sheet['quartets'] = {}

    for SRR_accession, num_reads in SRR_accessions.loc[screen_name]['num_reads'].iteritems():
        sample_sheet['quartets'][SRR_accession] = {which: f'{SRR_accession}_{which[-1]}' for which in ['R1', 'R2']}
        sample_sheet['quartets'][SRR_accession]['num_reads'] = num_reads

    return sample_sheet

def write_SRA_pool_sample_sheet(base_dir, screen_name):
    sample_sheet = load_SRA_pool_sample_sheet(screen_name)

    base_dir = Path(base_dir)

    results_dir = base_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    screen_dir = results_dir / screen_name
    screen_dir.mkdir(exist_ok=True)

    sample_sheet_fn = screen_dir / 'sample_sheet.yaml'
    sample_sheet_fn.write_text(yaml.safe_dump(sample_sheet))

def chunk_number_to_string(chunk_number):
    return f'{chunk_number:06d}'

class FastqChunker:
    def __init__(self, base_dir, group, quartet_name, which, reads_per_chunk=None, queue=None, debug=False, from_SRA=False):
        self.base_dir = Path(base_dir)
        self.group = group
        self.quartet_name = quartet_name
        self.which = which
        self.reads_per_chunk = reads_per_chunk
        self.queue = queue
        self.debug = debug

        if from_SRA:
            sample_sheet = load_SRA_pool_sample_sheet(self.group)
        else:
            sample_sheet = load_sample_sheet(self.base_dir, self.group)

        quartet = sample_sheet['quartets'][quartet_name]
        
        self.fn_name = quartet[self.which]
        
        group_dir = self.base_dir / 'data' / self.group
        self.input_fn = (group_dir / self.fn_name).with_suffix('.fastq.gz')
        
        self.chunks_dir = group_dir / 'chunks'
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
            line_groups = itertools.islice(line_groups, int(1e6))
        
        for read_number, line_group in enumerate(line_groups):
            if read_number % self.reads_per_chunk == 0:
                self.start_next_chunk()
            
            for line in line_group:
                self.current_fh.write(line)
                
        self.close_current_chunk()
        
        self.queue.put(('chunk', self.quartet_name, self.which, 'DONE'))
                
class UMISorters:
    def __init__(self, base_dir, group, quartet_name, chunk_number, from_SRA):
        self.base_dir = Path(base_dir)
        
        self.from_SRA = from_SRA
        if from_SRA:
            self.screen_name = group
        else:
            sample_sheet = load_sample_sheet(base_dir, group)
            self.group_name = sample_sheet['group_name']
        
        self.chunk_string = f'{quartet_name}_{chunk_number_to_string(chunk_number)}'

        self.sorters = defaultdict(list)


    def __getitem__(self, key):
        return self.sorters[key]

    def sort_and_write(self):
        for sample, fixed_guide, variable_guide in sorted(self.sorters):
            reads = self.sorters[sample, fixed_guide, variable_guide]
            sorted_reads = sorted(reads, key=lambda r: r.name)

            if self.from_SRA:
                if sample != self.screen_name:
                    raise ValueError(sample, self.screen_name)
                pool_name = self.screen_name
            else:
                pool_name = f'{self.group_name}_{sample}'

            pool = repair_seq.pooled_screen.PooledScreen(self.base_dir, pool_name)

            exp = pool.single_guide_experiment(fixed_guide, variable_guide)
            output_dir = exp.fns['chunks']
            output_dir.mkdir(exist_ok=True, parents=True)
            
            fn = output_dir / f'{self.chunk_string}_R2.fastq.gz'
            
            with gzip.open(fn, 'wt', compresslevel=1) as zfh:
                for read in sorted_reads:
                    zfh.write(str(read))

            del self.sorters[sample, fixed_guide, variable_guide]
            del sorted_reads
            
def split_into_chunks(base_dir, group, quartet_name, which, reads_per_chunk, queue, debug, from_SRA):
    chunker = FastqChunker(base_dir, group, quartet_name, which, reads_per_chunk, queue, debug, from_SRA)
    return chunker.split_into_chunks()

def get_resolvers(base_dir, group, from_SRA):
    expected_seqs = {}
    resolvers = {}

    if from_SRA:
        sample_sheet = load_SRA_pool_sample_sheet(group)
    else:
        sample_sheet = load_sample_sheet(base_dir, group)

        sample_indices = {name: details['index'] for name, details in sample_sheet['pool_details'].items()}

        expected_indices = set()
        for seqs in sample_indices.values():
            if not isinstance(seqs, list):
                seqs = [seqs]
            expected_indices.update(seqs)

        expected_seqs['sample'] = expected_indices
        resolvers['sample'] = utilities.get_one_mismatch_resolver(sample_indices).get

    if 'fixed_guide_library' not in sample_sheet:
        # If there weren't multiple fixed guide pools present, keep everything
        # to allow possibility of outcomes that don't include the intended NotI site.
        def fixed_guide_barcode_resolver(*args):
            return {'none'}

        resolvers['fixed_guide_barcode'] = fixed_guide_barcode_resolver
        expected_seqs['fixed_guide_barcode'] = set()

    else:
        fixed_guide_library = repair_seq.guide_library.GuideLibrary(base_dir, sample_sheet['fixed_guide_library'])
        resolvers['fixed_guide_barcode'] = utilities.get_one_mismatch_resolver(fixed_guide_library.guide_barcodes).get
        expected_seqs['fixed_guide_barcode'] = set(fixed_guide_library.guide_barcodes)

    variable_guide_library = repair_seq.guide_library.GuideLibrary(base_dir, sample_sheet['variable_guide_library'])
    guides = fasta.to_dict(variable_guide_library.fns['guides_fasta'])
    resolvers['variable_guide'] = utilities.get_one_mismatch_resolver({g: s[:45] for g, s in guides.items()}).get
    expected_seqs['variable_guide'] = set(guides.values())

    return resolvers, expected_seqs

def demux_chunk_from_SRA(base_dir, screen_name, quartet_name, chunk_number, queue):
    resolvers, expected_seqs = get_resolvers(base_dir, screen_name, from_SRA=True)

    SRA_annotation = Annotations['SRA']
    Annotation = Annotations['UMI_guide']
    
    fastq_fns = [FastqChunker(base_dir, screen_name, quartet_name, which, from_SRA=True).get_chunk_fn(chunk_number) for which in ['R1', 'R2']]

    sorters = UMISorters(base_dir, screen_name, quartet_name, chunk_number, from_SRA=True)

    counts = defaultdict(Counter)

    sample_sheet = load_SRA_pool_sample_sheet(screen_name)

    guide_barcode_slice = idx[:22]

    if 'fixed_guide_library' in sample_sheet:
        # If a guide barcode is present, remove it from R2 before passing along
        # to simplify analysis of common sequences in pool.
        after_guide_barcode_slice = idx[22:] 
    else:
        after_guide_barcode_slice = idx[:]

    for R1, R2 in fastq.read_pairs(*fastq_fns, up_to_space=True):
        if R1.name != R2.name:
            raise ValueError('read pair out of sync')

        sample = screen_name

        variable_guides = resolvers['variable_guide'](R1.seq[:45], 'unknown')
        if len(variable_guides) == 1:
            variable_guide = next(iter(variable_guides))
        else:
            variable_guide = 'unknown'

        guide_barcode = R2.seq[guide_barcode_slice]
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

        incoming_annotation = SRA_annotation.from_identifier(R1.name)

        annotation = Annotation(guide=R1.seq,
                                guide_qual=fastq.sanitize_qual(R1.qual),
                                original_name=incoming_annotation['original_name'],
                                UMI=incoming_annotation['UMI_seq'],
                               )
        R2.name = str(annotation)

        sorters[sample, fixed_guide, variable_guide].append(R2[after_guide_barcode_slice])

    sorters.sort_and_write()

    chunk_dir = base_dir / 'data' / screen_name / 'chunks'
    for k, cs in counts.items():
        cs = pd.Series(cs).sort_values(ascending=False)
        fn = chunk_dir / f'{k}_{quartet_name}_{chunk_number_to_string(chunk_number)}.txt'
        cs.to_csv(fn, sep='\t', header=False)

    # Delete the chunk.
    for fastq_fn in fastq_fns:
        fastq_fn.unlink()

    queue.put(('demux', quartet_name, chunk_number))

def demux_chunk(base_dir, group, quartet_name, chunk_number, queue):
    resolvers, expected_seqs = get_resolvers(base_dir, group, from_SRA=False)
    
    Annotation = Annotations['UMI_guide']
    
    fastq_fns = [FastqChunker(base_dir, group, quartet_name, which).get_chunk_fn(chunk_number) for which in fastq.quartet_order]

    sorters = UMISorters(base_dir, group, quartet_name, chunk_number, from_SRA=False)

    counts = defaultdict(Counter)

    sample_sheet = load_sample_sheet(base_dir, group)

    guide_barcode_slice = idx[:22]

    if 'fixed_guide_library' in sample_sheet:
        # If a guide barcode is present, remove it from R2 before passing along
        # to simplify analysis of common sequences in pool.
        after_guide_barcode_slice = idx[22:] 
    else:
        after_guide_barcode_slice = idx[:]

    for quartet in fastq.read_quartets(fastq_fns, up_to_space=True):
        if len({r.name for r in quartet}) != 1:
            raise ValueError('quartet out of sync')

        samples = resolvers['sample'](quartet.I2.seq, {'unknown'})

        if len(samples) == 1:
            sample = next(iter(samples))
        else:
            sample = 'unknown'

        counts['sample'][quartet.I2.seq] += 1

        variable_guides = resolvers['variable_guide'](quartet.R1.seq[:45], 'unknown')
        if len(variable_guides) == 1:
            variable_guide = next(iter(variable_guides))
        else:
            variable_guide = 'unknown'

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

        original_name = quartet.R1.name
        UMI = quartet.I1.seq
        guide_seq = quartet.R1.seq
        guide_qual = fastq.sanitize_qual(quartet.R1.qual)

        annotation = Annotation(guide=guide_seq,
                                guide_qual=guide_qual,
                                original_name=original_name,
                                UMI=UMI,
                               )
        quartet.R2.name = str(annotation)

        sorters[sample, fixed_guide, variable_guide].append(quartet.R2[after_guide_barcode_slice])

    sorters.sort_and_write()

    chunk_dir = base_dir / 'data' / group / 'chunks'
    for k, cs in counts.items():
        cs = pd.Series(cs).sort_values(ascending=False)
        fn = chunk_dir / f'{k}_{quartet_name}_{chunk_number_to_string(chunk_number)}.txt'
        cs.to_csv(fn, sep='\t', header=False)

    # Delete the chunk.
    for fastq_fn in fastq_fns:
        fastq_fn.unlink()

    queue.put(('demux', quartet_name, chunk_number))

def merge_seq_counts(base_dir, group, k, from_SRA):
    chunk_dir = Path(base_dir) / 'data' / group / 'chunks'
    count_fns = sorted(chunk_dir.glob(f'{k}*.txt'))

    counts = Counter()
    for fn in count_fns:
        for line in open(fn):
            seq, count = line.strip().split()
            counts[seq] += int(count)

    merged_fn = Path(base_dir) / 'data' / group / f'{k}_stats.txt'

    resolvers, expected_seqs = get_resolvers(base_dir, group, from_SRA)
    resolver = resolvers[k]
    expected_seqs = expected_seqs[k]

    with open(merged_fn, 'w') as fh:
        total = sum(counts.values())

        for seq, count in counts.most_common(100):
            name = resolver(seq, '')

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
    chunk_dir = Path(base_dir) / 'data' / group / 'chunks'
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

def demux_group(base_dir,
                group,
                debug=False,
                reads_per_chunk=int(5e6),
                from_SRA=False,
                just_chunk=False,
               ):
    '''
    just_chunk: Only split input files into chunks (don't demux them). 
    '''
    logging.info(f'Demultiplexing {group} in {base_dir}')

    if from_SRA:
        write_SRA_pool_sample_sheet(base_dir, group)
        sample_sheet = load_SRA_pool_sample_sheet(group)
        relevant_read_types = ['R1', 'R2']
        demux_chunk_function = demux_chunk_from_SRA
    else:
        make_pool_sample_sheets(base_dir, group)
        sample_sheet = load_sample_sheet(base_dir, group)
        relevant_read_types = fastq.quartet_order
        demux_chunk_function = demux_chunk

    quartet_names = sorted(sample_sheet['quartets'])

    if debug:
        reads_per_chunk = int(5e5)

    chunks_per_quartet = [int(np.ceil(d['num_reads'] / reads_per_chunk)) for q, d in sample_sheet['quartets'].items()]
    total_chunks = sum(chunks_per_quartet)

    manager = multiprocessing.Manager()
    tasks_done_queue = manager.Queue()
    
    chunks_done = defaultdict(set) 
    
    chunk_pool = multiprocessing.Pool(processes=4 * len(quartet_names))
    chunk_progress = tqdm.tqdm(desc='Chunk progress', total=total_chunks)
    chunk_results = []

    demux_pool = multiprocessing.Pool(processes=4 * len(quartet_names))
    demux_progress = tqdm.tqdm(desc='Demux progress', total=total_chunks)
    demux_results = []
    
    with chunk_pool, demux_pool:

        unfinished_chunkers = set()
        
        for quartet_name in quartet_names:
            for which in relevant_read_types:
                args = (base_dir, group, quartet_name, which, reads_per_chunk, tasks_done_queue, debug, from_SRA)
                chunk_result = chunk_pool.apply_async(split_into_chunks, args)

                if debug:
                    result = chunk_result.get()
                    if not chunk_result.successful():
                        print(result)

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
                    if chunks_done[quartet_name, chunk_number] == set(relevant_read_types):
                        chunk_progress.update()
                        
                        if not just_chunk:
                            args = (base_dir, group, quartet_name, chunk_number, tasks_done_queue)
                            demux_result = demux_pool.apply_async(demux_chunk_function, args)

                            if debug:
                                result = demux_result.get()
                                if not demux_result.successful():
                                    print(result)

                                demux_results.append(demux_result)

            elif task_type == 'demux':
                demux_progress.update()

        if not just_chunk:
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

    if not just_chunk:
        demux_progress.close()

        logging.info('Merging statistics...')

        merge_pool = multiprocessing.Pool(processes=3)
        merge_results = []
        if not from_SRA:
            merge_results.append(merge_pool.apply_async(merge_seq_counts, args=(base_dir, group, 'sample', from_SRA)))
        if 'fixed_guide_library' in sample_sheet:
            merge_results.append(merge_pool.apply_async(merge_seq_counts, args=(base_dir, group, 'fixed_guide_barcode', from_SRA)))

        merge_results.append(merge_pool.apply_async(merge_ids, args=(base_dir, group)))

        merge_pool.close()
        merge_pool.join()

        for merge_result in merge_results:
            if not merge_result.successful():
                print(merge_result.get())

        chunk_dir = base_dir / 'data' / group / 'chunks'
        shutil.rmtree(str(chunk_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', type=Path, default=Path.home() / 'projects' / 'repair_seq')
    parser.add_argument('group_name')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--from_SRA', action='store_true')
    parser.add_argument('--just_chunk', action='store_true')
    parser.add_argument('--reads_per_chunk', type=int, default=int(5e6))

    args = parser.parse_args()

    demux_group(args.base_dir, args.group_name,
                debug=args.debug,
                from_SRA=args.from_SRA,
                reads_per_chunk=args.reads_per_chunk,
                just_chunk=args.just_chunk,
               )