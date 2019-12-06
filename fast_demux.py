import argparse
import gzip
import itertools
import multiprocessing

from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml
import tqdm

from hits import fastq, fasta, utilities
import ddr.collapse
import ddr.guide_library

def load_sample_sheet(base_dir):
    sample_sheet_fn = Path(base_dir)  / 'sample_sheet.yaml'
    sample_sheet = yaml.safe_load(sample_sheet_fn.read_text())
    return sample_sheet
    
def chunk_number_to_string(chunk_number):
    return f'{chunk_number:06d}'

class FastqChunker():
    def __init__(self, base_dir, quartet_name, which, reads_per_chunk=None, queue=None):
        self.base_dir = Path(base_dir)
        self.quartet_name = quartet_name
        self.which = which
        self.reads_per_chunk = reads_per_chunk
        self.queue = queue
        
        sample_sheet = load_sample_sheet(self.base_dir)
        quartet = sample_sheet['quartets'][quartet_name]
        
        self.fn_name = quartet[self.which]
        
        self.input_fn = (self.base_dir / self.fn_name).with_suffix('.fastq.gz')
        
        self.chunks_dir = self.base_dir / 'chunks'
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
        
        for read_number, line_group in enumerate(line_groups):
            if read_number % self.reads_per_chunk == 0:
                self.start_next_chunk()
            
            for line in line_group:
                self.current_fh.write(line)
                
        self.close_current_chunk()
        
        if self.queue is not None:
            self.queue.put(('chunk', self.quartet_name, self.which, 'DONE'))
                
class UMISorters():
    def __init__(self, base_dir, quartet_name, chunk_number):
        self.base_dir = Path(base_dir)
        
        sample_sheet_fn = self.base_dir  / 'sample_sheet.yaml'
        sample_sheet = yaml.safe_load(sample_sheet_fn.read_text())
        self.group_name = sample_sheet['group_name']
        
        self.chunk_string = f'{quartet_name}_{chunk_number_to_string(chunk_number)}'

        self.sorters = defaultdict(list)

    def __getitem__(self, key):
        return self.sorters[key]

    def sort_and_write(self):
        for sample, fixed_guide, variable_guide in sorted(self.sorters):
            reads = self.sorters[sample, fixed_guide, variable_guide]
            sorted_reads = sorted(reads, key=lambda r: r.name)

            output_dir = self.base_dir.parent / f'{self.group_name}_{sample}' / 'by_guide' / self.chunk_string
            output_dir.mkdir(exist_ok=True, parents=True)
            
            fn = output_dir / f'{fixed_guide}-{variable_guide}_R2.fastq.gz'
            
            with gzip.open(fn, 'wt', compresslevel=1) as zfh:
                for read in sorted_reads:
                    zfh.write(str(read))

            del self.sorters[sample, fixed_guide, variable_guide]
            del sorted_reads
            
def split_into_chunks(base_dir, quartet_name, which, reads_per_chunk, queue):
    chunker = FastqChunker(base_dir, quartet_name, which, reads_per_chunk, queue)
    chunker.split_into_chunks()

def demux_chunk(base_dir, quartet_name, chunk_number, queue):
    sample_sheet = load_sample_sheet(base_dir)

    sample_indices = sample_sheet['sample_indices']
    sample_index_resolver = utilities.get_one_mismatch_resolver(sample_indices).get

    if 'fixed_guide_barcodes' not in sample_sheet:
        # If there weren't multiple fixed guide pools present, keep everythin
        # to allow possibility of outcomes that don't include the intended NotI site.
        def fixed_guide_barcode_resolver(*args):
            return 'none'
    else:
        fixed_guide_barcodes = sample_sheet['fixed_guide_barcodes']
        fixed_guide_barcode_resolver = utilities.get_one_mismatch_resolver(fixed_guide_barcodes).get

    R2_guide_barcode_slice = slice(6, 14)

    guide_library = ddr.guide_library.GuideLibrary('/home/jah/projects/ddr', sample_sheet['guide_library'])
    guides = fasta.to_dict(guide_library.fns['guides_fasta'])
    variable_guide_resolver = utilities.get_one_mismatch_resolver({g: s[:45] for g, s in guides.items()}).get
    
    Annotation = ddr.collapse.Annotations['UMI_guide']
    
    fastq_fns = [FastqChunker(base_dir, quartet_name, which).get_chunk_fn(chunk_number) for which in fastq.quartet_order]

    sorters = UMISorters(base_dir, quartet_name, chunk_number)

    for quartet in fastq.read_quartets(fastq_fns, up_to_space=True):
        if len({r.name for r in quartet}) != 1:
            raise ValueError('quartet out of sync')

        sample = sample_index_resolver(quartet.I2.seq, 'unknown')

        variable_guide = variable_guide_resolver(quartet.R1.seq, 'unknown')

        guide_barcode = quartet.R2.seq[R2_guide_barcode_slice]
        fixed_guide = fixed_guide_barcode_resolver(guide_barcode, 'unknown')

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

        sorters[sample, fixed_guide, variable_guide].append(quartet.R2)

    sorters.sort_and_write()

    # Delete the chunk.
    for fastq_fn in fastq_fns:
        fastq_fn.unlink()
            
    queue.put(('demux', quartet_name, chunk_number))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', type=Path, default=Path.home() / 'projects' / 'ddr')
    parser.add_argument('group_name')

    args = parser.parse_args()
    
    group_dir = args.base_dir / 'data' / args.group_name

    sample_sheet = load_sample_sheet(group_dir)
    
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
    demux_pool = multiprocessing.Pool(processes=5 * len(quartet_names))
    
    with chunk_pool, demux_pool:
        chunk_results = []
        demux_results = []

        unfinished_chunkers = set()
        
        for quartet_name in quartet_names:
            for which in fastq.quartet_order:
                args = (group_dir, quartet_name, which, reads_per_chunk, tasks_done_queue)
                chunk_result = chunk_pool.apply_async(split_into_chunks, args)
                #print(chunk_result.get())
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
                        
                        args = (group_dir, quartet_name, chunk_number, tasks_done_queue)
                        demux_result = demux_pool.apply_async(demux_chunk, args)
                        demux_results.append(demux_result)

            elif task_type == 'demux':
                demux_progress.update()
        
        while demux_progress.n < chunk_progress.n:
            task_type, *task_info = tasks_done_queue.get()
            if task_type == 'demux':
                demux_progress.update()
            else:
                raise ValueError(task_type)
        
        chunk_pool.join()
        for chunk_result in chunk_results:
            if not chunk_result.successful():
                print(chunk_result.get())
                
        demux_pool.close()
                
        demux_pool.join()
        for demux_result in demux_results:
            if not demux_result.successful():
                print(demux_result.get())
